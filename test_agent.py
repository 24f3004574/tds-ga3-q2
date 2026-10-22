"""Protocol smoke test. Usage: python test_agent.py https://your-app.vercel.app/a2a/
Sends a fake 12-package batch (costs a fraction of a cent of model usage)."""
import json, sys, uuid, concurrent.futures as cf
import httpx

BASE = sys.argv[1].rstrip("/")
H = lambda tok: {"Authorization": f"Bearer {tok}", "A2A-Version": "1.0", "Content-Type": "application/a2a+json"}
A, B = "test-user-" + uuid.uuid4().hex, "test-user-" + uuid.uuid4().hex
ok = lambda name, cond: print(("PASS " if cond else "FAIL ") + name)

def batch_msg(mid):
    pk = [{"packageId": f"pkg-{i}", "documents": [
        f"Cover sheet [COVER-{i}]",
        f"Invoice INV-{1000+i} from Acme Supplies for INR {500+i}.00. Within autonomous authority, reconciled. "
        f"[REC-{i}-A] [REC-{i}-B] [REC-{i}-C]"] } for i in range(12)]
    return {"message": {"messageId": mid, "role": "ROLE_USER", "parts": [{
        "mediaType": "application/vnd.ga5.invoice-claim-batch+json",
        "data": {"batchId": "b-" + mid, "policyRevision": "r1", "packages": pk}}]},
        "configuration": {"returnImmediately": False}}

c = httpx.Client(timeout=60)
ok("card public", c.get(BASE.rsplit("/", 1)[0] + "/.well-known/agent-card.json").status_code == 200)
ok("no auth -> 401/403", c.get(f"{BASE}/tasks").status_code in (401, 403))
ok("bad version -> 400", c.get(f"{BASE}/tasks", headers={**H(A), "A2A-Version": "0.3"}).status_code == 400)

mid = "m-" + uuid.uuid4().hex
body = batch_msg(mid)
with cf.ThreadPoolExecutor(2) as ex:   # concurrent equivalent requests
    r1, r2 = list(ex.map(lambda b: c.post(f"{BASE}/message:send", headers=H(A), json=b),
                         [body, {**body, "configuration": {"returnImmediately": True}}]))
ok("send 200", r1.status_code == 200 and r2.status_code == 200)
t1, t2 = r1.json()["task"], r2.json()["task"]
ok("same task for concurrent dupes", t1["id"] == t2["id"])
ok("INPUT_REQUIRED", t1["status"]["state"] == "TASK_STATE_INPUT_REQUIRED")
props = t1["artifacts"][0]["parts"][0]["data"]["proposals"]
ok("12 proposals, unique actionIds", len(props) == 12 and len({p["actionId"] for p in props}) == 12)
ok("conflict -> 409", c.post(f"{BASE}/message:send", headers=H(A),
   json={**body, "message": {**body["message"], "parts": body["message"]["parts"][:1] and [{**body["message"]["parts"][0], "data": {**body["message"]["parts"][0]["data"], "policyRevision": "r2"}}]}}).status_code == 409)

tid = t1["id"]
ok("outsider GET 403/404", c.get(f"{BASE}/tasks/{tid}", headers=H(B)).status_code in (403, 404))
ok("outsider list empty", c.get(f"{BASE}/tasks", headers=H(B)).json() == {"tasks": []})
ok("outsider cancel 403/404", c.post(f"{BASE}/tasks/{tid}:cancel", headers=H(B)).status_code in (403, 404))

results = [{"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
            "outcome": "ACCEPTED" if i % 2 == 0 else "REJECTED", "receiptNonce": "nonce-" + uuid.uuid4().hex}
           for i, p in enumerate(props)]
cont = {"message": {"messageId": "m-" + uuid.uuid4().hex, "taskId": tid, "contextId": t1["contextId"], "role": "ROLE_USER",
        "parts": [{"mediaType": "application/vnd.ga5.invoice-action-results+json",
                   "data": {"batchId": "b-" + mid, "results": results}}]}}
bad = json.loads(json.dumps(cont)); bad["message"]["messageId"] = "m-" + uuid.uuid4().hex
bad["message"]["parts"][0]["data"]["results"][0]["actionId"] = "wrong-action-id-xxxx"
ok("mismatched continuation -> 409", c.post(f"{BASE}/message:send", headers=H(A), json=bad).status_code == 409)
rc = c.post(f"{BASE}/message:send", headers=H(A), json=cont)
ok("continuation 200", rc.status_code == 200)
done = rc.json()["task"]
ok("COMPLETED", done["status"]["state"] == "TASK_STATE_COMPLETED")
ex_ = done["artifacts"][-1]["parts"][0]["data"]["executions"]
ok("receipts = accepted only", len(ex_) == 6)
ok("replay same continuation same task", c.post(f"{BASE}/message:send", headers=H(A), json=cont).json()["task"] == done)
ok("cancel after terminal -> 409", c.post(f"{BASE}/tasks/{tid}:cancel", headers=H(A)).status_code == 409)
ok("GET task unchanged", c.get(f"{BASE}/tasks/{tid}", headers=H(A)).json() == done)
lst = c.get(f"{BASE}/tasks", headers=H(A))
ok("list ok and < 512KiB", lst.status_code == 200 and len(lst.content) < 512 * 1024)
print("Note: this only tests protocol behavior, not decision accuracy.")
