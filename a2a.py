import hashlib
import json
import os
import re
import time
import traceback
import uuid
from datetime import datetime, timezone
from decimal import Decimal
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
CACHE_VERSION = os.getenv("CACHE_VERSION", "v2")
NO_MINOR_UNIT_CURRENCIES = {"JPY", "KRW", "VND"}
COVER_PATTERN = re.compile(
    r"Supplier (?P<vendor>.+?); invoice (?P<invoice>\S+?); stated total (?P<currency>[A-Z]{3}) (?P<amount>[\d,]+(?:\.\d+)?)"
)


def model_name():
    return os.getenv("AIPIPE_MODEL", "gpt-4.1-mini")


def ai_client():
    token = os.getenv("AIPIPE_TOKEN")
    if not token:
        raise HTTPException(status_code=503, detail="AI service is not configured")
    return OpenAI(
        base_url=os.getenv("AI_BASE_URL", "https://aipipe.org/openai/v1"),
        api_key=token,
        max_retries=0,
    )


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


def money_text(currency, amount_minor):
    return f"{currency} {amount_minor // 100}.{amount_minor % 100:02d}"


def build_rationale(action, facts, refs, reason):
    reason = " ".join((reason or "").split())[:700]
    text = (
        f"Chosen action: {action} for invoice {facts['invoiceNumber']} from {facts['vendorName']} "
        f"({money_text(facts['currency'], facts['amountMinor'])}). "
        f"The controlling evidence {refs[0]}, {refs[1]} and {refs[2]} supports this action."
    )
    if reason:
        text += f" {reason}"
    return text[:1500]


# ---- fast path: decisive paragraph is found in code, the model only picks the action
def fast_info(package):
    """Return {refs, paragraph, facts} when the package has exactly one paragraph with
    three distinct bracketed refs and a parseable cover line; otherwise None."""
    docs = package.get("documents") if isinstance(package, dict) else None
    if not isinstance(docs, list) or not docs or not all(isinstance(d, dict) and isinstance(d.get("text"), str) for d in docs):
        return None
    candidates = []
    for doc in docs:
        for paragraph in re.split(r"\n\s*\n", doc["text"]):
            refs = list(dict.fromkeys(REF_PATTERN.findall(paragraph)))
            if len(refs) == 3:
                candidates.append((refs, " ".join(paragraph.split())))
    if len(candidates) != 1:
        return None
    full_text = "\n\n".join(d["text"] for d in docs)
    match = COVER_PATTERN.search(full_text)
    if not match:
        return None
    currency = match["currency"]
    if currency in NO_MINOR_UNIT_CURRENCIES:
        return None
    try:
        amount_minor = int((Decimal(match["amount"].replace(",", "")) * 100).to_integral_value())
    except Exception:
        return None
    return {
        "refs": candidates[0][0],
        "paragraph": candidates[0][1],
        "facts": {
            "vendorName": match["vendor"].strip(),
            "invoiceNumber": match["invoice"].strip(),
            "amountMinor": amount_minor,
            "currency": currency,
        },
    }


FAST_PROMPT = """You are an accounts-payable decision agent. For each case you get ONE controlling paragraph (already isolated from noise) and the stated invoice total. Choose exactly one action from that paragraph alone.

Actions:
- settle_invoice: the records reconcile with no exception, goods are received, and no earlier payment exists for THIS invoice, and nothing in the paragraph pauses or escalates it. A prior payment to the same supplier for the same amount but a DIFFERENT invoice does NOT make a duplicate: settle.
- request_approval: records reconcile cleanly, but the payable total is above this queue's autonomous authority / delegation ceiling, so a named approver is required.
- hold_invoice: payment is paused until a stated verification finishes, typically a changed bank account / beneficiary / remittance instruction whose callback or out-of-band check is not complete.
- reject_duplicate: the same commercial invoice (same supplier, invoice identity, currency, amount; differences are only OCR/punctuation/another scan) was already paid, so a second payment must be rejected.
- open_exception: material records conflict: total outside tolerance, invoice does not reconcile with order/receipt valuation, incompatible explanations, no corrected document; needs an exception workflow.

Decide from what the paragraph actually establishes, not from words that merely appear. If the paragraph raises one specific blocking issue, choose the action for that issue; if it only confirms a clean reconciliation with no issue, choose settle_invoice. When it compares a ceiling with the total, compare the numbers yourself.

Return JSON only: {"decisions":[{"packageId":"...","action":"...","reason":"one or two plain sentences (max 45 words) naming the facts in the paragraph that lead to the action"}]} with exactly one decision per packageId."""


def call_fast_model(items, timeout):
    payload = [
        {
            "packageId": pid,
            "statedTotal": money_text(info["facts"]["currency"], info["facts"]["amountMinor"]),
            "controllingParagraph": info["paragraph"],
        }
        for pid, info in items
    ]
    completion = ai_client().with_options(timeout=timeout).chat.completions.create(
        model=model_name(),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": FAST_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
        ],
    )
    parsed = json.loads(completion.choices[0].message.content or "{}")
    decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
    if not isinstance(decisions, list):
        raise ValueError("Fast model response did not contain decisions")
    return decisions


def fast_decision(info, item):
    action = item.get("action") if isinstance(item, dict) else None
    if action not in ACTIONS:
        raise ValueError("Unsupported action")
    reason = item.get("reason") if isinstance(item.get("reason"), str) else ""
    return {
        "action": action,
        "facts": dict(info["facts"]),
        "evidenceRefs": list(info["refs"]),
        "rationale": build_rationale(action, info["facts"], info["refs"], reason),
    }


# ---- full path (fallback): the model reads the whole package
def normalize_decision(item, expected_package_id, package):
    """Validate a full-package model proposal. Rationale is built in code."""
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

    clean_facts = {
        "vendorName": vendor.strip(),
        "invoiceNumber": invoice.strip(),
        "amountMinor": amount_minor,
        "currency": currency.upper(),
    }
    return {
        "packageId": expected_package_id,
        "action": action,
        "facts": clean_facts,
        "evidenceRefs": refs,
        "rationale": build_rationale(action, clean_facts, refs, item.get("rationale") if isinstance(item.get("rationale"), str) else ""),
    }


SYSTEM_PROMPT = """You are an evidence-based invoice action agent for accounts payable.
You receive several invoice packages. Each package mixes decisive records with noise: cover sheets, archived/old examples, training decoys, negated statements and irrelevant action words. Decide from the single paragraph that actually governs THIS invoice (it contains exactly three bracketed references), not from keywords elsewhere.

Choose exactly one action per package:
- settle_invoice: valid, reconciled, within autonomous authority, no prior payment of THIS invoice (a prior payment of a DIFFERENT invoice is not a duplicate).
- request_approval: commercially valid but the total is above the queue's delegated authority.
- hold_invoice: payment pauses until a stated verification completes (e.g. changed bank details with the callback pending).
- reject_duplicate: the same commercial invoice was already paid.
- open_exception: material records conflict and need an exception workflow.

Evidence: cite EXACTLY the three bracketed references of the controlling paragraph, copied character-for-character with brackets. Never cite cover-sheet, archive or training references.
Facts: vendorName and invoiceNumber as on the signed cover sheet; amountMinor is an INTEGER in minor units (1,234.50 -> 123450); currency is the 3-letter ISO code.
Rationale: one or two plain sentences on why this action applies.

Return JSON only: {"proposals":[{"packageId":"...","action":"...","facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"INR"},"evidenceRefs":["[..]","[..]","[..]"],"rationale":"..."}]} with exactly one proposal per supplied packageId and no others."""


def call_model(packages, timeout):
    payload = [{"packageId": package_id(pkg), "package": canonical_package(pkg)} for pkg in packages]
    completion = ai_client().with_options(timeout=timeout).chat.completions.create(
        model=model_name(),
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
    hashes = {pid: json_hash({"cache": CACHE_VERSION, "package": content}) for pid, content in canonical.items()}
    cached_by_hash = cache_get(set(hashes.values()))
    decisions = {pid: cached_by_hash[h] for pid, h in hashes.items() if h in cached_by_hash}

    pending = [pkg for pkg in packages if package_id(pkg) not in decisions]
    for attempt in range(2):
        if not pending:
            break
        pending_by_id = {package_id(pkg): pkg for pkg in pending}
        new_entries = []

        # Attempt 0 uses the fast path where possible; the retry reads whole packages.
        fast_items = []
        if attempt == 0:
            for pid, pkg in pending_by_id.items():
                info = fast_info(pkg)
                if info:
                    fast_items.append((pid, info))
        fast_ids = {pid for pid, _ in fast_items}

        if fast_items and deadline - time.monotonic() >= 8:
            try:
                results = call_fast_model(fast_items, timeout=min(30.0, deadline - time.monotonic() - 3))
                info_by_id = dict(fast_items)
                for item in results:
                    pid = item.get("packageId") if isinstance(item, dict) else None
                    if pid not in info_by_id or pid in decisions:
                        continue
                    try:
                        decisions[pid] = fast_decision(info_by_id[pid], item)
                    except ValueError as exc:
                        print("FAST REJECTED", pid, exc)
                        continue
                    new_entries.append((hashes[pid], canonical[pid], decisions[pid]))
            except HTTPException:
                raise
            except Exception:
                traceback.print_exc()

        slow = [pkg for pid, pkg in pending_by_id.items() if pid not in fast_ids and pid not in decisions]
        if attempt == 1:
            slow = [pkg for pid, pkg in pending_by_id.items() if pid not in decisions]
        if slow and deadline - time.monotonic() >= 8:
            slow_by_id = {package_id(pkg): pkg for pkg in slow}
            try:
                items = call_model(slow, timeout=min(30.0, deadline - time.monotonic() - 3))
                for item in items:
                    pid = item.get("packageId") if isinstance(item, dict) else None
                    if pid not in slow_by_id or pid in decisions:
                        continue
                    try:
                        decision = normalize_decision(item, pid, slow_by_id[pid])
                    except ValueError as exc:
                        print("REJECTED", pid, exc, json.dumps(item)[:300])
                        continue
                    decisions[pid] = decision
                    new_entries.append((hashes[pid], canonical[pid], decision))
            except HTTPException:
                raise
            except Exception:
                traceback.print_exc()

        try:
            cache_put(new_entries)
        except Exception:
            traceback.print_exc()  # caching is an optimization; never fail the request over it
        pending = [pkg for pkg in pending if package_id(pkg) not in decisions]

    if pending:
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
