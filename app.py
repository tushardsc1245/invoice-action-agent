#!/usr/bin/env python3
"""
Invoice action agent - A2A 1.0 HTTP+JSON.

Layers (kept separate on purpose):
  1. Protocol : auth, version, media types, routes, envelopes
  2. Storage  : SQLite tasks, message idempotency, decision cache
  3. AI       : one batched LLM call per uncached group of packages

Run (single worker is REQUIRED - locks are in-process):
  pip install fastapi uvicorn
  export BASE_URL="https://your-host/a2a/"      # exactly what you submit to the grader
  export LLM_API_KEY="..."                       # any OpenAI-compatible provider
  uvicorn invoice_agent:app --host 0.0.0.0 --port 8000 --workers 1 --proxy-headers
"""
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

# ----------------------------------------------------------------- config
BASE_URL = os.environ.get("BASE_URL", "").strip()
BASE_PATH = urlparse(BASE_URL).path if BASE_URL else os.environ.get("BASE_PATH", "/a2a/")
ROUTE_PREFIX = BASE_PATH.rstrip("/")

# Optional allow-list. Empty (default) = every non-empty Bearer token is a separate user.
ALLOWED_TOKENS = {t.strip() for t in os.environ.get("ALLOWED_TOKENS", "").split(",") if t.strip()}

# Any OpenAI-compatible chat-completions endpoint (Gemini, Groq, OpenRouter, Ollama ...)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")
# Comma-separated fallback chain: if one model is overloaded (503) the next one is tried.
LLM_MODELS = [m.strip() for m in os.environ.get(
    "LLM_MODELS", LLM_MODEL + ",gemini-3.1-flash-lite,gemini-flash-lite-latest").split(",") if m.strip()]
# Packages per model call. Small chunks run in parallel, so each call is fast.
LLM_CHUNK = max(1, int(os.environ.get("LLM_CHUNK", "3")))

# Flip to "1" if the grader's feedback says evidence refs don't match and the
# documents show refs like "[ABC-1]" that you should return WITH the brackets.
EVIDENCE_WITH_BRACKETS = os.environ.get("EVIDENCE_WITH_BRACKETS", "0") == "1"

DB_PATH = os.environ.get("DB_PATH", "agent.db")
PROMPT_VERSION = "v1"
REQUEST_BUDGET_SECONDS = 42.0  # grader allows 45s per request

A2A_JSON = "application/a2a+json"
BATCH_MT = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MT = "application/vnd.ga5.invoice-action-proposals+json"
RESULTS_MT = "application/vnd.ga5.invoice-action-results+json"
RECEIPTS_MT = "application/vnd.ga5.invoice-action-receipts+json"

ACTIONS = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]

S_INPUT = "TASK_STATE_INPUT_REQUIRED"
S_DONE = "TASK_STATE_COMPLETED"
S_CANCELED = "TASK_STATE_CANCELED"
TERMINAL = {S_DONE, S_CANCELED, "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}

MAX_RESPONSE_BYTES = 480_000


# ----------------------------------------------------------------- helpers
def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def jresp(obj, status=200):
    body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return Response(body, status_code=status, media_type=A2A_JSON)


def err(status, code, message):
    # Generic bodies only: never echo ids or reveal existence.
    return jresp({"error": {"code": status, "status": code, "message": message}}, status)


def log(*a):
    print("[agent]", *a, flush=True)


# ----------------------------------------------------------------- storage
_db = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
_db.execute("PRAGMA journal_mode=WAL")
_dblock = threading.RLock()
with _dblock:
    _db.execute(
        "CREATE TABLE IF NOT EXISTS tasks (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE,"
        " principal TEXT, state TEXT, json TEXT)"
    )
    _db.execute(
        "CREATE TABLE IF NOT EXISTS messages (principal TEXT, message_id TEXT, hash TEXT, task_id TEXT,"
        " PRIMARY KEY (principal, message_id))"
    )
    _db.execute("CREATE TABLE IF NOT EXISTS cache (content_hash TEXT PRIMARY KEY, decision TEXT)")

_locks = {}
_locks_meta = threading.Lock()


def lock_for(key):
    with _locks_meta:
        return _locks.setdefault(key, threading.Lock())


def load_task(principal, task_id):
    with _dblock:
        r = _db.execute("SELECT json FROM tasks WHERE id=? AND principal=?", (task_id, principal)).fetchone()
    return json.loads(r[0]) if r else None


def save_task(principal, task, message_id=None, message_hash=None, new=False):
    """Atomically persist the task and (optionally) the message-id record."""
    with _dblock:
        _db.execute("BEGIN IMMEDIATE")
        try:
            if new:
                _db.execute(
                    "INSERT INTO tasks (id, principal, state, json) VALUES (?,?,?,?)",
                    (task["id"], principal, task["status"]["state"], json.dumps(task, ensure_ascii=False)),
                )
            else:
                _db.execute(
                    "UPDATE tasks SET state=?, json=? WHERE id=? AND principal=?",
                    (task["status"]["state"], json.dumps(task, ensure_ascii=False), task["id"], principal),
                )
            if message_id is not None:
                _db.execute(
                    "INSERT INTO messages (principal, message_id, hash, task_id) VALUES (?,?,?,?)",
                    (principal, message_id, message_hash, task["id"]),
                )
            _db.execute("COMMIT")
        except Exception:
            _db.execute("ROLLBACK")
            raise


def get_message_record(principal, mid):
    with _dblock:
        return _db.execute(
            "SELECT hash, task_id FROM messages WHERE principal=? AND message_id=?", (principal, mid)
        ).fetchone()


def cache_get(h):
    with _dblock:
        r = _db.execute("SELECT decision FROM cache WHERE content_hash=?", (h,)).fetchone()
    return json.loads(r[0]) if r else None


def cache_put(h, d):
    with _dblock:
        _db.execute("INSERT OR REPLACE INTO cache (content_hash, decision) VALUES (?,?)", (h, json.dumps(d)))


# ----------------------------------------------------------------- AI layer
class LLMFailure(Exception):
    pass


SYSTEM_PROMPT = """You are the decision engine of an accounts-payable agent. All data is synthetic.
For EACH invoice package choose exactly one action:

- settle_invoice: invoice is valid, reconciled (matches PO / receipt / vendor records) and within autonomous payment authority.
- request_approval: commercially valid, but the amount or type is outside delegated authority, so a higher approver must sign off.
- hold_invoice: payment pauses until a specifically stated verification completes (e.g. bank-detail callback, tax/ID check, goods receipt, document confirmation).
- reject_duplicate: the SAME commercial invoice (same vendor and invoice identity) was ALREADY PAID. A payment for a DIFFERENT invoice - even same vendor or same amount - is NOT a duplicate.
- open_exception: material records conflict (invoice vs PO vs receipt vs vendor master etc.) and need an exception workflow.

Documents deliberately mix useful facts with noise. IGNORE: cover-sheet summaries, archive/old/historical examples,
training examples and decoys, negated statements (e.g. "this is not a duplicate", "no hold is required"),
and action words used in irrelevant contexts. Decide ONLY from the current, authoritative finding about THIS invoice.
That finding is normally one paragraph carrying exactly three bracketed references.

For every package return:
- packageId (copied exactly)
- action (one of the five above)
- vendorName, invoiceNumber, currency (ISO code) and amountMinor of the invoice being decided (not of an archive example).
  amountMinor is an INTEGER in minor units (e.g. INR 1,234.50 -> 123450; zero-decimal currencies keep the whole amount).
- evidenceRefs: exactly THREE references taken from the decisive paragraph. {REF_RULE}
  Do NOT include the cover-sheet reference, archive examples or decoy references.
- reason: 1-2 sentences (max 350 characters) explaining why that action follows from the decisive paragraph.
  Do not name any action other than the chosen one.

Output ONLY one JSON object, no markdown:
{"decisions":[{"packageId":"...","action":"...","vendorName":"...","invoiceNumber":"...","currency":"INR","amountMinor":12345,"evidenceRefs":["...","...","..."],"reason":"..."}]}
Return one decision per package, in the same order."""


def build_system_prompt():
    if EVIDENCE_WITH_BRACKETS:
        rule = "Copy each reference exactly as written INCLUDING its square brackets."
    else:
        rule = "Copy the text inside each pair of square brackets exactly, WITHOUT the brackets."
    return SYSTEM_PROMPT.replace("{REF_RULE}", rule)


def call_llm(system, user, timeout, model=None):
    url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
    base = {
        "model": model or LLM_MODEL,
        "temperature": 0,
        "max_tokens": 4000,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + LLM_API_KEY,
        "User-Agent": "invoice-agent/1.0",
    }
    for use_json_mode in (True, False):
        payload = dict(base)
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 400 and use_json_mode:
                continue  # provider may not support response_format
            raise
    raise LLMFailure("no response")


def parse_json_loose(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e < s:
        raise ValueError("no json")
    return json.loads(text[s : e + 1])


def flat_strings(o, out):
    if isinstance(o, str):
        out.append(o)
    elif isinstance(o, dict):
        for k, v in o.items():
            out.append(str(k))
            flat_strings(v, out)
    elif isinstance(o, list):
        for v in o:
            flat_strings(v, out)
    return out


BRACKET_RE = re.compile(r"\[([^\[\]\n]{1,120})\]")


def paragraph_units(pkg):
    """Paragraph-like text units of a package (split on blank lines)."""
    units = []
    for s in flat_strings(pkg, []):
        for u in re.split(r"\n\s*\n", s):
            if u.strip():
                units.append(u)
    return units


def distinct_refs(unit):
    seen, out = set(), []
    for m in BRACKET_RE.findall(unit):
        m = m.strip()
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def norm_ref(r):
    r = str(r).strip()
    if r.startswith("[") and r.endswith("]"):
        r = r[1:-1].strip()
    return r


def fmt_ref(r):
    return "[" + r + "]" if EVIDENCE_WITH_BRACKETS else r


def pick_refs(pkg, model_refs):
    """Validate that refs are real and come from one paragraph; fall back to a heuristic."""
    units = paragraph_units(pkg)
    refs = []
    for r in model_refs or []:
        n = norm_ref(r)
        if n and n not in refs:
            refs.append(n)
    if len(refs) == 3:
        for u in units:
            present = set(distinct_refs(u))
            if all(r in present for r in refs):
                return [fmt_ref(r) for r in refs]
    # Heuristic: paragraph with exactly three distinct bracketed references.
    cands = [distinct_refs(u) for u in units]
    cands = [c for c in cands if len(c) == 3]
    for c in cands:  # prefer the candidate that overlaps most with the model's answer
        pass
    if cands:
        best = max(cands, key=lambda c: len(set(c) & set(refs)))
        return [fmt_ref(r) for r in best]
    allrefs = set()
    for u in units:
        allrefs.update(distinct_refs(u))
    good = [r for r in refs if r in allrefs][:3]
    return [fmt_ref(r) for r in good]


def to_minor(v):
    if isinstance(v, bool):
        raise ValueError
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if v == int(v) else int(round(v * 100))
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        if re.fullmatch(r"-?\d+\.\d+", s):
            return int(round(float(s) * 100))
    raise ValueError("amount")


def validate_decision(pkg, d):
    try:
        if not isinstance(d, dict) or d.get("action") not in ACTIONS:
            return None
        vendor = str(d.get("vendorName", "")).strip()
        inv = str(d.get("invoiceNumber", "")).strip()
        cur = str(d.get("currency", "")).strip().upper()
        if not (vendor and inv and cur):
            return None
        amount = to_minor(d.get("amountMinor"))
        if amount < 0:
            return None
        refs = pick_refs(pkg, d.get("evidenceRefs"))
        if len(refs) < 2:
            return None
        reason = re.sub(r"\s+", " ", str(d.get("reason", ""))).strip()[:600]
        return {
            "action": d["action"],
            "facts": {"vendorName": vendor, "invoiceNumber": inv, "amountMinor": amount, "currency": cur},
            "evidenceRefs": refs,
            "reason": reason,
        }
    except Exception:
        return None


def pkg_hash(pkg):
    body = {k: v for k, v in pkg.items() if k != "packageId"}
    return sha(PROMPT_VERSION + "|" + canon(body))


def _decide_chunk(pkgs, idxs, policy_revision, deadline, results, lock):
    """Decide one small group of packages; retry on overload, rotating through fallback models."""
    todo = list(idxs)
    attempt = 0
    while todo:
        remaining = deadline - time.time()
        if remaining < 6:
            return
        model = LLM_MODELS[attempt % len(LLM_MODELS)]
        attempt += 1
        parts = [f"Policy revision: {policy_revision}\nDecide every package below.\n"]
        for i in todo:
            parts.append(f"=== PACKAGE {pkgs[i]['packageId']} ===\n" + json.dumps(pkgs[i], ensure_ascii=False))
        try:
            raw = call_llm(build_system_prompt(), "\n".join(parts), timeout=min(remaining - 2, 30), model=model)
            decs = parse_json_loose(raw).get("decisions", [])
        except urllib.error.HTTPError as e:
            log(f"llm http error {e.code} model={model}")
            time.sleep(min(0.5 * attempt, 2.0))
            continue
        except Exception as e:  # noqa
            log(f"llm error model={model}:", repr(e)[:200])
            continue
        by_id = {str(d.get("packageId")): d for d in decs if isinstance(d, dict)}
        for i in list(todo):
            v = validate_decision(pkgs[i], by_id.get(pkgs[i]["packageId"]))
            if v:
                cache_put(pkg_hash(pkgs[i]), v)
                with lock:
                    results[i] = v
                todo.remove(i)


def decide_packages(pkgs, policy_revision, deadline):
    results = {}
    missing = []
    for i, p in enumerate(pkgs):
        c = cache_get(pkg_hash(p))
        if c:
            results[i] = c
        else:
            missing.append(i)

    if missing:
        lock = threading.Lock()
        threads = []
        for k in range(0, len(missing), LLM_CHUNK):
            chunk = missing[k : k + LLM_CHUNK]
            t = threading.Thread(
                target=_decide_chunk, args=(pkgs, chunk, policy_revision, deadline, results, lock), daemon=True
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join(max(deadline - time.time(), 0.1))

    undecided = [i for i in range(len(pkgs)) if i not in results]
    if undecided:
        raise LLMFailure(f"{len(undecided)} packages undecided")
    return [results[i] for i in range(len(pkgs))]


def build_rationale(action, facts, reason, refs):
    text = (
        f"Recommended action {action} for invoice {facts['invoiceNumber']} from {facts['vendorName']}. "
        f"{reason} Decisive evidence: {', '.join(refs)}."
    )
    if len(text) > 1500:
        keep = 1500 - (len(text) - len(reason))
        reason = reason[: max(keep, 0)]
        text = (
            f"Recommended action {action} for invoice {facts['invoiceNumber']} from {facts['vendorName']}. "
            f"{reason} Decisive evidence: {', '.join(refs)}."
        )[:1500]
    return text


# ----------------------------------------------------------------- protocol core
def is_terminal(task):
    return task["status"]["state"] in TERMINAL


def new_task(principal, msg, h, mid, start):
    part = next((p for p in msg["parts"] if p.get("mediaType") == BATCH_MT and isinstance(p.get("data"), dict)), None)
    if part is None:
        return err(400, "INVALID_ARGUMENT", "Unsupported message content.")
    data = part["data"]
    batch_id, pkgs = data.get("batchId"), data.get("packages")
    if not isinstance(batch_id, str) or not batch_id or not isinstance(pkgs, list) or not pkgs:
        return err(400, "INVALID_ARGUMENT", "Invalid batch.")
    ids = []
    for p in pkgs:
        if not isinstance(p, dict) or not isinstance(p.get("packageId"), str) or not p["packageId"]:
            return err(400, "INVALID_ARGUMENT", "Invalid package.")
        ids.append(p["packageId"])
    if len(set(ids)) != len(ids):
        return err(400, "INVALID_ARGUMENT", "Duplicate package.")

    try:
        with open("last_batch.json", "w", encoding="utf-8") as f:  # handy for debugging (synthetic data)
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass

    try:
        decisions = decide_packages(pkgs, data.get("policyRevision", ""), start + REQUEST_BUDGET_SECONDS)
    except LLMFailure as e:
        log("decision failure:", e)
        return err(503, "UNAVAILABLE", "Temporarily unable to process the request.")

    proposals = []
    for p, d in zip(pkgs, decisions):
        action_id = "act_" + sha(f"{batch_id}|{p['packageId']}|{pkg_hash(p)}")[:28]
        proposals.append(
            {
                "packageId": p["packageId"],
                "actionId": action_id,
                "action": d["action"],
                "facts": d["facts"],
                "evidenceRefs": d["evidenceRefs"],
                "rationale": build_rationale(d["action"], d["facts"], d["reason"], d["evidenceRefs"]),
            }
        )
        log("decided", p["packageId"], d["action"], d["evidenceRefs"])

    task_id = str(uuid.uuid4())
    context_id = msg["contextId"] if isinstance(msg.get("contextId"), str) and msg["contextId"] else str(uuid.uuid4())
    stored = dict(msg)
    stored["taskId"] = task_id
    stored["contextId"] = context_id
    task = {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": S_INPUT, "timestamp": now_iso()},
        "artifacts": [
            {
                "artifactId": "proposals-" + task_id,
                "name": "invoice-action-proposals",
                "parts": [{"mediaType": PROPOSALS_MT, "data": {"batchId": batch_id, "proposals": proposals}}],
            }
        ],
        "history": [stored],
    }
    save_task(principal, task, mid, h, new=True)  # state stored BEFORE responding
    return jresp({"task": task})


def continue_task(principal, msg, h, mid):
    tid = msg.get("taskId")
    if not isinstance(tid, str) or not tid:
        return err(400, "INVALID_ARGUMENT", "Invalid task reference.")
    with lock_for(("task", tid)):
        task = load_task(principal, tid)
        if task is None:
            return err(404, "NOT_FOUND", "Task not found.")
        if msg.get("contextId") != task["contextId"]:
            return err(400, "INVALID_ARGUMENT", "Context mismatch.")
        if is_terminal(task) or task["status"]["state"] != S_INPUT:
            return err(409, "FAILED_PRECONDITION", "Task cannot accept this message.")

        rparts = [p for p in msg["parts"] if p.get("mediaType") == RESULTS_MT and isinstance(p.get("data"), dict)]
        if len(rparts) != 1:
            return err(400, "INVALID_ARGUMENT", "Invalid results.")
        rdata = rparts[0]["data"]
        pdata = task["artifacts"][0]["parts"][0]["data"]
        if rdata.get("batchId") != pdata["batchId"]:
            return err(400, "INVALID_ARGUMENT", "Batch mismatch.")
        results = rdata.get("results")
        if not isinstance(results, list) or not results:
            return err(400, "INVALID_ARGUMENT", "Invalid results.")

        props = {p["packageId"]: p for p in pdata["proposals"]}
        seen, accepted = set(), {}
        for r in results:
            if not isinstance(r, dict):
                return err(400, "INVALID_ARGUMENT", "Invalid result.")
            pid = r.get("packageId")
            prop = props.get(pid) if isinstance(pid, str) else None
            if prop is None or pid in seen:
                return err(400, "INVALID_ARGUMENT", "Result mismatch.")
            seen.add(pid)
            if r.get("actionId") != prop["actionId"] or r.get("action") != prop["action"]:
                return err(400, "INVALID_ARGUMENT", "Result mismatch.")
            outcome = r.get("outcome")
            if outcome not in ("ACCEPTED", "REJECTED"):
                return err(400, "INVALID_ARGUMENT", "Invalid outcome.")
            if outcome == "ACCEPTED":
                nonce = r.get("receiptNonce")
                if not isinstance(nonce, str) or not nonce.strip():
                    return err(400, "INVALID_ARGUMENT", "Missing receipt.")
                accepted[pid] = nonce

        executions = []
        for p in pdata["proposals"]:  # only accepted, in proposal order; matches stored proposal exactly
            if p["packageId"] in accepted:
                executions.append(
                    {
                        "packageId": p["packageId"],
                        "actionId": p["actionId"],
                        "action": p["action"],
                        "receiptNonce": accepted[p["packageId"]],
                        "facts": p["facts"],
                        "evidenceRefs": p["evidenceRefs"],
                    }
                )

        stored = dict(msg)
        stored["taskId"] = tid
        stored["contextId"] = task["contextId"]
        task["history"].append(stored)
        task["artifacts"].append(
            {
                "artifactId": "receipts-" + tid,
                "name": "invoice-action-receipts",
                "parts": [{"mediaType": RECEIPTS_MT, "data": {"batchId": pdata["batchId"], "executions": executions}}],
            }
        )
        task["status"] = {"state": S_DONE, "timestamp": now_iso()}
        save_task(principal, task, mid, h)
        return jresp({"task": task})


def handle_send(principal, raw, start):
    try:
        body = json.loads(raw)
    except Exception:
        return err(400, "INVALID_ARGUMENT", "Malformed JSON.")
    if not isinstance(body, dict) or not isinstance(body.get("message"), dict):
        return err(400, "INVALID_ARGUMENT", "Missing message.")
    msg = body["message"]
    mid = msg.get("messageId")
    if not isinstance(mid, str) or not mid.strip():
        return err(400, "INVALID_ARGUMENT", "Invalid messageId.")
    if msg.get("role") != "ROLE_USER":
        return err(400, "INVALID_ARGUMENT", "Invalid role.")
    parts = msg.get("parts")
    if not isinstance(parts, list) or not parts or not all(isinstance(p, dict) for p in parts):
        return err(400, "INVALID_ARGUMENT", "Invalid parts.")

    h = sha(canon(msg))  # message only; configuration ignored; keys sorted recursively
    with lock_for(("msg", principal, mid)):
        rec = get_message_record(principal, mid)
        if rec:
            if rec[0] != h:
                return jresp(
                    {
                        "error": {
                            "code": 409,
                            "status": "IDEMPOTENCY_CONFLICT",
                            "message": "IDEMPOTENCY_CONFLICT: messageId was already used with different content.",
                            "details": [{"reason": "IDEMPOTENCY_CONFLICT"}],
                        }
                    },
                    409,
                )
            task = load_task(principal, rec[1])
            if task is None:
                return err(404, "NOT_FOUND", "Task not found.")
            return jresp({"task": task})
        if msg.get("taskId") is not None:
            return continue_task(principal, msg, h, mid)
        return new_task(principal, msg, h, mid, start)


def do_cancel(principal, tid):
    with lock_for(("task", tid)):
        task = load_task(principal, tid)
        if task is None:
            return err(404, "NOT_FOUND", "Task not found.")
        if is_terminal(task):
            return err(409, "FAILED_PRECONDITION", "Task cannot be canceled.")
        task["status"] = {"state": S_CANCELED, "timestamp": now_iso()}
        save_task(principal, task)
        return jresp(task)


def trim_history(task, n):
    t = dict(task)
    hist = t.get("history", [])
    if n is None:
        return t
    if n <= 0:
        t.pop("history", None)
    else:
        t["history"] = hist[-n:]
    return t


def parse_int(v):
    try:
        return int(v)
    except Exception:
        return None


# ----------------------------------------------------------------- HTTP surface
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
router = APIRouter(prefix=ROUTE_PREFIX)


def preflight(request: Request):
    auth = request.headers.get("authorization", "")
    parts = auth.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        return None, err(401, "UNAUTHENTICATED", "Authentication required.")
    token = parts[1].strip()
    if ALLOWED_TOKENS and token not in ALLOWED_TOKENS:
        return None, err(401, "UNAUTHENTICATED", "Authentication required.")
    if request.headers.get("a2a-version", "").strip() != "1.0":
        return None, err(400, "VERSION_NOT_SUPPORTED", "Unsupported protocol version.")
    return sha("token|" + token), None


def media_ok(request: Request):
    ct = request.headers.get("content-type", "").split(";")[0].strip().lower()
    return ct == A2A_JSON


@app.get("/.well-known/agent-card.json")
async def agent_card(request: Request):
    base = BASE_URL
    if not base:
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        base = f"{proto}://{request.headers.get('host', request.url.netloc)}{BASE_PATH}"
    card = {
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one evidence-cited business action per invoice, "
        "and executes only grader-accepted actions with idempotent task state.",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "supportedInterfaces": [{"url": base, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "defaultInputModes": [BATCH_MT, RESULTS_MT],
        "defaultOutputModes": [PROPOSALS_MT, RECEIPTS_MT],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice action decisions",
                "description": "Chooses settle, approval, hold, duplicate-rejection or exception handling "
                "for each invoice package and cites the decisive evidence.",
                "tags": ["invoice", "accounts-payable", "approval", "duplicate-detection"],
            }
        ],
    }
    return Response(json.dumps(card).encode(), media_type="application/json")


@app.get("/healthz")
async def healthz():
    return Response(b"ok", media_type="text/plain")


@router.post("/message:send")
async def message_send(request: Request):
    start = time.time()
    principal, e = preflight(request)
    if e:
        return e
    if not media_ok(request):
        return err(415, "UNSUPPORTED_MEDIA_TYPE", "Unsupported media type.")
    raw = await request.body()
    return await run_in_threadpool(handle_send, principal, raw, start)


@router.post("/tasks/{task_id}:cancel")
async def task_cancel(task_id: str, request: Request):
    principal, e = preflight(request)
    if e:
        return e
    return await run_in_threadpool(do_cancel, principal, task_id)


@router.get("/tasks/{task_id}")
async def task_get(task_id: str, request: Request):
    principal, e = preflight(request)
    if e:
        return e
    task = await run_in_threadpool(load_task, principal, task_id)
    if task is None:
        return err(404, "NOT_FOUND", "Task not found.")
    n = parse_int(request.query_params.get("historyLength"))
    return jresp(trim_history(task, n))


@router.get("/tasks")
async def task_list(request: Request):
    principal, e = preflight(request)
    if e:
        return e
    q = request.query_params
    n = parse_int(q.get("historyLength"))
    page_size = min(max(parse_int(q.get("pageSize")) or 50, 1), 100)
    offset = parse_int(q.get("pageToken")) or 0

    def work():
        with _dblock:
            rows = _db.execute(
                "SELECT json FROM tasks WHERE principal=? ORDER BY seq DESC", (principal,)
            ).fetchall()
        tasks = [json.loads(r[0]) for r in rows]
        if q.get("contextId"):
            tasks = [t for t in tasks if t["contextId"] == q.get("contextId")]
        if q.get("status"):
            tasks = [t for t in tasks if t["status"]["state"] == q.get("status")]
        total = len(tasks)
        out, size, idx = [], 0, offset
        while idx < total and len(out) < page_size:
            t = trim_history(tasks[idx], n if n is not None else 0)  # list omits history by default
            s = len(json.dumps(t, ensure_ascii=False))
            if out and size + s > MAX_RESPONSE_BYTES:
                break
            out.append(t)
            size += s
            idx += 1
        body = {"tasks": out}
        if idx < total:
            body["nextPageToken"] = str(idx)
        return jresp(body)

    return await run_in_threadpool(work)


app.include_router(router)
