import re

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Load local .env before importing modules that read environment variables.
load_dotenv()

from a2a import router as a2a_router  # noqa: E402

# Only the Agent Card is public. Disable FastAPI's docs/OpenAPI routes.
app = FastAPI(
    title="A2A Invoice Agent",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.include_router(a2a_router)

BASE_URL = "https://tds-ga3-q2-plum.vercel.app/a2a/"  # must match what you submit exactly

DEFAULT_CODES = {
    400: "INVALID_REQUEST",
    401: "UNAUTHENTICATED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    502: "UPSTREAM_FAILED",
    503: "UNAVAILABLE",
}


def error_response(status_code: int, detail: str):
    code = detail if re.fullmatch(r"[A-Z][A-Z_]+", detail or "") else DEFAULT_CODES.get(status_code, "ERROR")
    message = "Idempotency conflict" if code == "IDEMPOTENCY_CONFLICT" else (detail or "Error")
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
        media_type="application/a2a+json",
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return error_response(exc.status_code, str(exc.detail))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Generic body: never leak internals or other users' identifiers.
    return error_response(500, "Internal error")


@app.get("/.well-known/agent-card.json")
def agent_card():
    card = {
        "name": "A2A Invoice Action Agent",
        "description": (
            "An invoice decision agent that proposes evidence-based "
            "actions and executes only accepted proposals."
        ),
        "version": "1.0.0",
        "capabilities": {},
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",
                "description": (
                    "Evaluates invoice packages and proposes safe actions "
                    "with supporting evidence."
                ),
                "tags": ["invoice", "approval", "evidence", "payments"],
            }
        ],
        "supportedInterfaces": [
            {
                "url": BASE_URL,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ],
        "defaultInputModes": ["application/vnd.ga5.invoice-claim-batch+json"],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json",
        ],
    }
    return JSONResponse(content=card, media_type="application/json")
