import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
from psycopg.types.json import Jsonb
from starlette.concurrency import run_in_threadpool

from database import connect_db

router = APIRouter(prefix="/a2a")

BATCH_MEDIA = "application/vnd.ga5.invoice-claim-batch+json"
RESULTS_MEDIA = "application/vnd.ga5.invoice-action-results+json"
PROPOSALS_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"
A2A_MEDIA = "application/a2a+json"
ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}
REF_PATTERN = re.compile(r"\[[^\[\]]+\]")
MAX_BODY = 512 * 1024
LIST_BUDGET = 380 * 1024      # keep GET /tasks safely under the 512 KiB cap
REQUEST_BUDGET_SECONDS = 38   # each grader request gets 45 s


# ---------------------------------------------------------------- helpers
def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def encoded_size(data) -> int:
    return len(json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))


def response(data, status_code=200, media_type=A2A_MEDIA):
    if encoded_size(data) > MAX_BODY:
        raise HTTPException(status_code=500, detail="Response exceeds protocol size limit")
    return JSONResponse(content=data, status_code=status_code, media_type=media_type)


async def authenticated_principal(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    parts = authorization.split()
    if len(parts) != 2 or parts[0] != "Bearer" or not parts[1]:
        raise HTTPException(status_code=401, detail="Authentication required")

    if request.headers.get("A2A-Version") != "1.0":
        raise HTTPException(status_code=400, detail="Unsupported A2A version")

    if request.method in {"POST", "PUT", "PATCH"}:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        has_body = request.headers.get("content-length", "0") not in {"", "0"}
        # Bodyless POSTs (e.g. cancel) are fine; any body must be A2A JSON.
        if (content_type or has_body) and content_type != A2A_MEDIA:
            raise HTTPException(status_code=400, detail="Unsupported media type")

    return hashlib.sha256(parts[1].encode("utf-8")).hexdigest()


def get_message(body):
    message = body.get("message") if isinstance(body, dict) else None
    if not isinstance(message, dict):
        raise HTTPException(status_code=400, detail="Invalid A2A message")
    message_id = message.get("messageId")
    if not isinstance(message_id, str) or not message_id.strip():
        raise HTTPException(status_code=400, detail="Invalid message ID")
    if message.get("role") != "ROLE_USER":
        raise HTTPException(status_code=400, detail="Invalid message role")
    if not isinstance(message.get("parts"), list) or not message["parts"]:
        raise HTTPException(status_code=400, detail="Invalid message parts")
    return message


def find_data_part(message, media_type):
    for part in message.get("parts", []):
        if isinstance(part, dict) and part.get("mediaType") == media_type and isinstance(part.get("data"), dict):
            return part["data"]
    return None


def add_message_metadata(message, task_id, context_id):
    # A2A 1.0 ProtoJSON has no "kind" discriminator, so we do not add one.
    result = dict(message)
    result.setdefault("taskId", task_id)
    result.setdefault("contextId", context_id)
    return result


def make_artifact(media_type, data, name):
    return {
        "artifactId": str(uuid.uuid4()),
        "name": name,
        "parts": [{"mediaType": media_type, "data": data}],
    }


def canonical_package(package):
    if not isinstance(package, dict):
        raise HTTPException(status_code=400, detail="Invalid invoice package")
    return {key: value for key, value in package.items() if key != "packageId"}


def package_id(package):
    value = package.get("packageId") if isinstance(package, dict) else None
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=400, detail="Invoice package is missing packageId")
    return value


# ---------------------------------------------------------------- AI layer
def ai_client():
    token = os.getenv("AIPIPE_TOKEN")
    if not token:
        raise HTTPException(status_code=503, detail="AI service is not configured")
    return OpenAI(base_url="https://aipipe.org/openrouter/v1", api_key=token, max_retries=0)


def coerce_amount(value):
    if isinstance(value, bool):
        raise ValueError("Model returned an invalid amount")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        value = int(value.strip())
    if not isinstance(value, int) or value < 0:
        raise ValueError("Model returned an invalid amount")
    return value


def normalize_decision(item, expected_package_id, package):
    """Validate a model proposal. The rationale is built in code so it always
    names the action and cites evidence; the model only chooses action, facts
    and evidence references."""
    if not isinstance(item, dict):
        raise ValueError("Model returned an invalid proposal")

    action = item.get("action")
    if action not in ACTIONS:
        raise ValueError("Model returned an unsupported action")

    facts = item.get("facts")
    if not isinstance(facts, dict):
        raise ValueError("Model returned invalid facts")
    vendor = facts.get("vendorName")
    invoice = facts.get("invoiceNumber")
    currency = facts.get("currency")
    if not isinstance(vendor, str) or not vendor.strip():
        raise ValueError("Model returned an invalid vendor name")
    if isinstance(invoice, (int, float)) and not isinstance(invoice, bool):
        invoice = str(invoice)
    if not isinstance(invoice, str) or not invoice.strip():
        raise ValueError("Model returned an invalid invoice number")
    if not isinstance(currency, str) or not re.fullmatch(r"[A-Za-z]{3}", currency):
        raise ValueError("Model returned an invalid currency")
    amount_minor = coerce_amount(facts.get("amountMinor"))

    refs = item.get("evidenceRefs")
    if not isinstance(refs, list) or len(refs) != 3:
        raise ValueError("Model must return exactly three evidence references")
    if any(not isinstance(ref, str) or not REF_PATTERN.fullmatch(ref.strip()) for ref in refs):
        raise ValueError("Model returned a malformed evidence reference")
    refs = [ref.strip() for ref in refs]
    if len(set(refs)) != 3:
        raise ValueError("Evidence references must be unique")

    package_text = json.dumps(package, ensure_ascii=False, separators=(",", ":"))
    if any(ref not in package_text for ref in refs):
        raise ValueError("Model cited evidence absent from the invoice package")

    vendor = vendor.strip()
    invoice = invoice.strip()
    model_reason = item.get("rationale") if isinstance(item.get("rationale"), str) else ""
    model_reason = " ".join(model_reason.split())[:900]
    rationale = (
        f"Chosen action: {action} for invoice {invoice} from {vendor}. "
        f"The decisive records {refs[0]}, {refs[1]} and {refs[2]} support this action."
    )
    if model_reason:
        rationale += f" {model_reason}"
    rationale = rationale[:1500]

    return {
        "packageId": expected_package_id,
        "action": action,
        "facts": {
            "vendorName": vendor,
            "invoiceNumber": invoice,
            "amountMinor": amount_minor,
            "currency": currency.upper(),
        },
        "evidenceRefs": refs,
        "rationale": rationale,
    }


SYSTEM_PROMPT = """You are an evidence-based invoice action agent for accounts payable.
You receive several invoice packages. Each package mixes decisive records with noise: cover sheets, archived/old examples, training decoys, negated statements ("this is NOT a duplicate", "no approval needed in the old policy") and irrelevant action words. Decide from the paragraph that actually governs THIS invoice under the current policy, not from keywords.

Choose exactly one action per package:
- settle_invoice: valid, reconciled, within autonomous authority.
- request_approval: commercially valid but outside delegated authority (amount above the approver's limit).
- hold_invoice: payment pauses until a stated verification completes (e.g. bank-detail callback, missing receipt/PO check).
- reject_duplicate: the SAME commercial invoice (same vendor, invoice number, amount) was already paid. A payment for a different invoice is NOT a duplicate.
- open_exception: material records conflict (amount, vendor, PO, currency mismatch) and need an exception workflow.

Evidence: cite EXACTLY three bracketed references, copied character-for-character (including the square brackets) from the single paragraph that determines the action. Never cite the cover-sheet reference, archive examples or training decoys.
Facts: vendorName, invoiceNumber as printed on the invoice; amountMinor is an INTEGER in minor units (paise for INR, cents for USD; 1,234.50 -> 123450); currency is the 3-letter ISO code.
Rationale: one or two plain sentences on why this action applies.

Return JSON only, shaped as {"proposals":[{"packageId":"...","action":"...","facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"INR"},"evidenceRefs":["[..]","[..]","[..]"],"rationale":"..."}]} with exactly one proposal per supplied packageId and no others."""


def call_model(packages, timeout):
    payload = [{"packageId": package_id(pkg), "package": canonical_package(pkg)} for pkg in packages]
    completion = ai_client().with_options(timeout=timeout).chat.completions.create(
        model=os.getenv("AIPIPE_MODEL", "openai/gpt-4.1-mini"),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
        ],
    )
    parsed = json.loads(completion.choices[0].message.content or "{}")
    items = parsed.get("proposals") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        raise ValueError("Model response did not contain proposals")
    return items


# ---------------------------------------------------------------- decision cache (own transactions)
def cache_get(hashes):
    if not hashes:
        return {}
    with connect_db() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT package_hash, decision_json FROM decision_cache WHERE package_hash = ANY(%s)",
                (list(hashes),),
            )
            return {row[0]: row[1] for row in cursor.fetchall()}


def cache_put(entries):
    """entries: list of (hash, canonical_package, decision). Committed immediately so
    good decisions survive even if another package in the batch fails."""
    if not entries:
        return
    with connect_db() as connection:
        with connection.cursor() as cursor:
            for digest, canonical, decision in entries:
                cursor.execute(
                    """INSERT INTO decision_cache (package_hash, canonical_package, decision_json)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (package_hash) DO UPDATE SET
                         decision_json = EXCLUDED.decision_json,
                         updated_at = CURRENT_TIMESTAMP""",
                    (digest, Jsonb(canonical), Jsonb(decision)),
                )


def build_proposals(batch, deadline):
    batch_id = batch.get("batchId")
    packages = batch.get("packages")
    policy_revision = batch.get("policyRevision")
    if (
        not isinstance(batch_id, str) or not batch_id.strip()
        or not isinstance(policy_revision, str) or not policy_revision.strip()
        or not isinstance(packages, list) or not packages
    ):
        raise HTTPException(status_code=400, detail="Invalid invoice batch")
    ids = [package_id(pkg) for pkg in packages]
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=400, detail="Duplicate package IDs")

    canonical = {package_id(pkg): canonical_package(pkg) for pkg in packages}
    hashes = {pid: json_hash(content) for pid, content in canonical.items()}
    cached_by_hash = cache_get(set(hashes.values()))
    decisions = {pid: cached_by_hash[h] for pid, h in hashes.items() if h in cached_by_hash}

    pending = [pkg for pkg in packages if package_id(pkg) not in decisions]
    for _attempt in range(2):  # one retry, only for packages that failed validation
        if not pending:
            break
        remaining = deadline - time.monotonic()
        if remaining < 8:
            break
        try:
            items = call_model(pending, timeout=min(30.0, remaining - 3))
        except HTTPException:
            raise
        except Exception:
            continue
        pending_by_id = {package_id(pkg): pkg for pkg in pending}
        fresh_entries = []
        for item in items:
            pid = item.get("packageId") if isinstance(item, dict) else None
            if pid not in pending_by_id or pid in decisions:
                continue
            try:
                decision = normalize_decision(item, pid, pending_by_id[pid])
            except ValueError:
                continue
            decisions[pid] = decision
            fresh_entries.append((hashes[pid], canonical[pid], decision))
        try:
            cache_put(fresh_entries)
        except Exception:
            pass  # caching is an optimization; never fail the request over it
        pending = [pkg for pkg in pending if package_id(pkg) not in decisions]

    if pending:
        # Successful decisions are already cached, so a retry only redoes these.
        raise HTTPException(status_code=502, detail="Invoice decision service failed")

    proposals = []
    action_ids = set()
    for pkg in packages:
        pid = package_id(pkg)
        decision = dict(decisions[pid])
        decision["packageId"] = pid
        action_id = uuid.uuid4().hex + uuid.uuid4().hex[:8]
        while action_id in action_ids:
            action_id = uuid.uuid4().hex + uuid.uuid4().hex[:8]
        action_ids.add(action_id)
        decision["actionId"] = action_id
        proposals.append(decision)
    return {"batchId": batch_id, "proposals": proposals}


# ---------------------------------------------------------------- task storage
def save_task(cursor, task, principal_hash, batch_id):
    cursor.execute(
        """INSERT INTO tasks (task_id, principal_hash, context_id, batch_id, state, task_json)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (task["id"], principal_hash, task["contextId"], batch_id, task["status"]["state"], Jsonb(task)),
    )


def save_dedup(cursor, principal_hash, message_id, message_hash, task):
    cursor.execute(
        """INSERT INTO message_dedup (principal_hash, message_id, message_hash, task_id, response_json)
           VALUES (%s, %s, %s, %s, %s)""",
        (principal_hash, message_id, message_hash, task["id"], Jsonb(task)),
    )


def process_initial(cursor, message, batch, principal_hash, deadline):
    proposal_data = build_proposals(batch, deadline)
    task_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now_iso()},
        "history": [add_message_metadata(message, task_id, context_id)],
        "artifacts": [make_artifact(PROPOSALS_MEDIA, proposal_data, "Invoice action proposals")],
    }
    save_task(cursor, task, principal_hash, batch["batchId"])
    return task


def process_continuation(cursor, message, results_data, principal_hash):
    task_id = message.get("taskId")
    context_id = message.get("contextId")
    if not isinstance(task_id, str) or not task_id or not isinstance(context_id, str) or not context_id:
        raise HTTPException(status_code=400, detail="Invalid task continuation")

    cursor.execute(
        "SELECT task_json, batch_id, state FROM tasks WHERE task_id = %s AND principal_hash = %s FOR UPDATE",
        (task_id, principal_hash),
    )
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    task, stored_batch_id, state = row
    if task.get("contextId") != context_id or results_data.get("batchId") != stored_batch_id:
        raise HTTPException(status_code=409, detail="Task continuation mismatch")
    if state != "TASK_STATE_INPUT_REQUIRED":
        raise HTTPException(status_code=409, detail="Task is already terminal")

    proposals_part = next(
        (part for artifact in task.get("artifacts", []) for part in artifact.get("parts", [])
         if part.get("mediaType") == PROPOSALS_MEDIA and isinstance(part.get("data"), dict)),
        None,
    )
    if not proposals_part:
        raise HTTPException(status_code=409, detail="Task has no proposal artifact")
    proposal_map = {p.get("packageId"): p for p in proposals_part["data"].get("proposals", [])}
    results = results_data.get("results")
    if not isinstance(results, list) or len(results) != len(proposal_map):
        raise HTTPException(status_code=400, detail="Invalid action results")

    seen = set()
    executions = []
    for item in results:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="Invalid action result")
        pid = item.get("packageId")
        proposal = proposal_map.get(pid)
        if not proposal or pid in seen:
            raise HTTPException(status_code=409, detail="Action result mismatch")
        seen.add(pid)
        if item.get("actionId") != proposal.get("actionId") or item.get("action") != proposal.get("action"):
            raise HTTPException(status_code=409, detail="Action result mismatch")
        if item.get("outcome") not in {"ACCEPTED", "REJECTED"}:
            raise HTTPException(status_code=400, detail="Invalid action outcome")
        nonce = item.get("receiptNonce")
        if not isinstance(nonce, str) or not nonce:
            raise HTTPException(status_code=400, detail="Invalid receipt nonce")
        if item["outcome"] == "ACCEPTED":
            executions.append({
                "packageId": pid,
                "actionId": proposal["actionId"],
                "action": proposal["action"],
                "receiptNonce": nonce,
                "facts": proposal["facts"],
                "evidenceRefs": proposal["evidenceRefs"],
            })
    if seen != set(proposal_map):
        raise HTTPException(status_code=400, detail="Action results are incomplete")

    task["history"].append(add_message_metadata(message, task_id, context_id))
    task["artifacts"].append(
        make_artifact(RECEIPTS_MEDIA, {"batchId": stored_batch_id, "executions": executions}, "Invoice action receipts")
    )
    task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now_iso()}
    cursor.execute(
        "UPDATE tasks SET state = %s, task_json = %s, updated_at = CURRENT_TIMESTAMP WHERE task_id = %s",
        ("TASK_STATE_COMPLETED", Jsonb(task), task_id),
    )
    return task


# ---------------------------------------------------------------- routes
def handle_send(principal_hash, body):
    deadline = time.monotonic() + REQUEST_BUDGET_SECONDS
    message = get_message(body)
    message_id = message["messageId"]
    message_hash = json_hash(message)  # message only; configuration is ignored
    is_continuation = bool(message.get("taskId"))

    try:
        with connect_db() as connection:
            with connection.cursor() as cursor:
                # Serialize same-principal retries before dedup check or any model call.
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (principal_hash + ":" + message_id,),
                )
                cursor.execute(
                    "SELECT message_hash, task_id FROM message_dedup WHERE principal_hash = %s AND message_id = %s",
                    (principal_hash, message_id),
                )
                prior = cursor.fetchone()
                if prior:
                    if prior[0] != message_hash:
                        raise HTTPException(status_code=409, detail="IDEMPOTENCY_CONFLICT")
                    cursor.execute(
                        "SELECT task_json FROM tasks WHERE task_id = %s AND principal_hash = %s",
                        (prior[1], principal_hash),
                    )
                    row = cursor.fetchone()
                    if not row:
                        raise HTTPException(status_code=404, detail="Task not found")
                    return response({"task": row[0]})

                if is_continuation:
                    results_data = find_data_part(message, RESULTS_MEDIA)
                    if results_data is None:
                        raise HTTPException(status_code=400, detail="Invalid result continuation")
                    task = process_continuation(cursor, message, results_data, principal_hash)
                else:
                    batch = find_data_part(message, BATCH_MEDIA)
                    if batch is None:
                        raise HTTPException(status_code=400, detail="Invalid invoice batch message")
                    task = process_initial(cursor, message, batch, principal_hash, deadline)

                save_dedup(cursor, principal_hash, message_id, message_hash, task)
                return response({"task": task})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Request could not be processed") from exc


@router.post("/message:send")
async def send_message(request: Request, principal_hash: str = Depends(authenticated_principal)):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    # Blocking DB/model work runs in a thread so polls and probes stay responsive.
    return await run_in_threadpool(handle_send, principal_hash, body)


def fit_task_list(tasks):
    """Keep GET /tasks under the response cap: full tasks if they fit, else
    history-free tasks, else only the newest tasks that fit (tasks are newest-first)."""
    if encoded_size({"tasks": tasks}) <= LIST_BUDGET:
        return tasks
    slim = [{**task, "history": []} for task in tasks]
    if encoded_size({"tasks": slim}) <= LIST_BUDGET:
        return slim
    kept, total = [], 2
    for task in slim:
        size = encoded_size(task) + 1
        if total + size > LIST_BUDGET:
            break
        kept.append(task)
        total += size
    return kept


@router.get("/tasks")
def list_tasks(principal_hash: str = Depends(authenticated_principal)):
    with connect_db() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT task_json FROM tasks WHERE principal_hash = %s ORDER BY created_at DESC LIMIT 200",
                (principal_hash,),
            )
            tasks = [row[0] for row in cursor.fetchall()]
    return response({"tasks": fit_task_list(tasks)})


@router.get("/tasks/{task_id}")
def get_task(task_id: str, principal_hash: str = Depends(authenticated_principal)):
    with connect_db() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT task_json FROM tasks WHERE task_id = %s AND principal_hash = %s",
                (task_id, principal_hash),
            )
            row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    return response(row[0])


@router.post("/tasks/{task_id}:cancel")
def cancel_task(task_id: str, principal_hash: str = Depends(authenticated_principal)):
    # The row lock serializes cancellation against a result continuation, which
    # locks the same row. Exactly one transition out of INPUT_REQUIRED wins.
    with connect_db() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT task_json, state FROM tasks WHERE task_id = %s AND principal_hash = %s FOR UPDATE",
                (task_id, principal_hash),
            )
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Task not found")
            task, state = row
            if state != "TASK_STATE_INPUT_REQUIRED":
                raise HTTPException(status_code=409, detail="Task is already terminal")

            task["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now_iso()}
            cursor.execute(
                """UPDATE tasks SET state = %s, task_json = %s, updated_at = CURRENT_TIMESTAMP
                   WHERE task_id = %s AND principal_hash = %s AND state = %s""",
                ("TASK_STATE_CANCELED", Jsonb(task), task_id, principal_hash, "TASK_STATE_INPUT_REQUIRED"),
            )
            if cursor.rowcount != 1:
                raise HTTPException(status_code=409, detail="Task is already terminal")
    return response(task)
