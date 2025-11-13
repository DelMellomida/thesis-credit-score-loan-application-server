from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.requests import Request
from app.api.auth_routes import router as auth_router
from app.api.loan_routes import router as loan_router
from app.api.document_routes import router as document_router
from app.api.model_routes import router as model_router
from app.api.reports_routes import router as reports_router
from contextlib import asynccontextmanager
from app.database.connection import init_db
from app.core.config import settings
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import logging
import traceback


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add critical security headers to all API responses.
    
    This middleware implements:
    1. X-Content-Type-Options: Prevents MIME type sniffing attacks
    2. X-Frame-Options: Legacy clickjacking protection (deprecated in favor of CSP)
    3. X-XSS-Protection: Legacy XSS protection header
    4. Referrer-Policy: Controls how referrer information is shared
    5. Permissions-Policy: Restricts access to sensitive APIs
    
    IMPORTANT: This middleware does NOT handle OPTIONS requests.
    OPTIONS requests are handled by CORSMiddleware which is registered first.
    If we intercept OPTIONS here, we bypass CORSMiddleware's CORS header injection,
    causing "No Access-Control-Allow-Origin header" errors in browsers.
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Let CORSMiddleware handle OPTIONS requests completely
        # Do NOT intercept OPTIONS here as it prevents CORS headers from being added
        response = await call_next(request)
        
        # Only add security headers to non-OPTIONS responses
        # OPTIONS responses are handled entirely by CORSMiddleware
        if request.method != "OPTIONS":
            # X-Content-Type-Options: Prevents MIME type sniffing
            response.headers["X-Content-Type-Options"] = "nosniff"
            
            # X-Frame-Options: Legacy clickjacking protection (for older browsers)
            # The modern alternative is CSP frame-ancestors directive
            response.headers["X-Frame-Options"] = "DENY"
            
            # X-XSS-Protection: Legacy XSS protection (for older browsers)
            response.headers["X-XSS-Protection"] = "1; mode=block"
            
            # Referrer-Policy: Only send referrer for same-origin requests
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            
            # Permissions-Policy: Disable access to sensitive browser APIs
            response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Note: CSP is not set here because:
        # 1. API responses should not have CSP directives
        # 2. CSP headers on API responses can interfere with CORS preflight handling
        # 3. The frontend CSP is properly handled by Next.js middleware
        
        return response


class DebugRequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Lightweight middleware to log incoming OPTIONS requests (and responses)
    for debugging CORS preflight issues coming from the browser.

    This middleware does NOT short-circuit requests. It simply logs the
    request headers and the response status/headers for paths under
    `/documents` when the HTTP method is OPTIONS. It is intentionally
    non-intrusive and can be removed after debugging.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        import logging

        logger = logging.getLogger("debug_middleware")

        # Only log OPTIONS requests for the documents API (where preflights occur)
        should_log = request.method == "OPTIONS" and request.url.path.startswith("/documents")

        if should_log:
            # Capture incoming headers as a dict for clearer logs
            try:
                incoming_headers = {k: v for k, v in request.headers.items()}
                # Redact any sensitive header values before logging
                sensitive = {"authorization", "cookie", "set-cookie", "x-api-key", "x-supabase-token", "idempotency-key", "supabase-service-role", "jwt-secret-key"}
                def _redact(hdr_dict):
                    out = {}
                    for kk, vv in hdr_dict.items():
                        if kk.lower() in sensitive:
                            out[kk] = "<redacted>"
                        else:
                            out[kk] = vv
                    return out
                incoming_headers = _redact(incoming_headers)
            except Exception:
                incoming_headers = {str(k): str(v) for k, v in request.scope.get("headers", [])}

            logger.warning(f"[DEBUG] Incoming OPTIONS {request.url.path} headers: {incoming_headers}")

        # Let the rest of the middleware stack (notably CORSMiddleware) handle the request
        response = await call_next(request)

        if should_log:
            try:
                resp_headers = dict(response.headers)
                # redact response headers that may contain sensitive tokens
                resp_headers = {k: ("<redacted>" if k.lower() in ("set-cookie", "authorization") else v) for k, v in resp_headers.items()}
            except Exception:
                resp_headers = {}
            logger.warning(f"[DEBUG] Response for OPTIONS {request.url.path}: status={response.status_code} headers={resp_headers}")

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(
    title="Credit Scoring and Loan Recommendation",
    description="Credit Scoring and Loan Recommendation",
    version="1.0.0",
    lifespan=lifespan 
)


# Global exception handlers to return structured JSON and log tracebacks
logger = logging.getLogger("server_exception_handler")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Return structured error payload for known HTTP exceptions
    body = {
        "error": {
            "code": getattr(exc, 'detail', 'http_error'),
            "message": str(exc.detail) if exc.detail else exc.status_code,
            "status_code": exc.status_code
        }
    }
    logger.warning(f"HTTPException handled: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content=body)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Validation errors from FastAPI/Pydantic
    body = {
        "error": {
            "code": "validation_error",
            "message": "Request validation failed",
            "details": exc.errors()
        }
    }
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(status_code=422, content=body)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Catch-all for unexpected exceptions. Log traceback and return a generic
    # structured error so clients receive consistent JSON.
    tb = traceback.format_exc()
    logger.error(f"Unhandled exception: {str(exc)}\n{tb}")
    body = {
        "error": {
            "code": "internal_server_error",
            "message": "An unexpected error occurred",
            "details": str(exc)
        }
    }
    return JSONResponse(status_code=500, content=body)

# CRITICAL: Add CORS middleware FIRST (middlewares execute in reverse order, so this runs LAST which is correct)
# Support comma-separated CLIENT_URL values (e.g. "http://localhost:3000,http://localhost:3001")
raw_origins = settings.CLIENT_URL or ""
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

# Always include both development ports for development environment
if not allowed_origins or any("localhost" in origin for origin in allowed_origins):
    allowed_origins = list(set(allowed_origins + ["http://localhost:3000", "http://localhost:3001"]))

print(f"DEBUG: CORS allowed origins: {allowed_origins}")

# Configure CORS with restrictive settings
# - Only allow specified origins (not wildcard *)
# - Allow credentials for authenticated requests
# - Restrict methods to necessary HTTP verbs
# - Restrict headers to necessary values
# NOTE: Middleware execution order in Starlette/FastAPI is LIFO (last added runs first).
# To ensure CORS preflight requests are handled by CORSMiddleware before any
# other middleware (so Access-Control-Allow-* headers are present), we add the
# SecurityHeadersMiddleware first and the CORSMiddleware last. That way CORSMiddleware
# executes first and can short-circuit/respond to OPTIONS preflight requests.
# Add Security Headers Middleware first (will execute after CORSMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# Configure CORS with restrictive settings and add it LAST so it executes FIRST
# and can properly handle preflight requests and inject Access-Control headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Specific origins only, not wildcard
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],  # Explicit HTTP methods
    # Allow common and custom headers that the frontend may send (e.g. x-nonce from Next middleware)
    allow_headers=[
        # Standard request headers
        "Content-Type",
        "Authorization",
        "Accept",
        "Origin",
        "Accept-Language",
        "Content-Language",

        # Common XHR / framework headers
        "X-Requested-With",
        "x-nonce",
        "X-Nonce",

        # Cache / validation headers that browsers include in preflight requests
        "Cache-Control",
        "Expires",
        "If-Modified-Since",
        "If-None-Match",
        "Pragma",
        "Surrogate-Control",
        # Idempotency header used by the client to safely retry uploads
        "Idempotency-Key",
    ],
    expose_headers=["Content-Type", "Authorization"],  # Headers that frontend can access
    max_age=3600,  # Cache CORS preflight for 1 hour
)

# Debug middleware: logs OPTIONS requests to /documents so we can see exactly
# what the browser sent and what the server returned (visible in container logs).
# It's added after CORSMiddleware so it executes before other middleware and
# can observe the raw incoming request. Remove this in production.
app.add_middleware(DebugRequestLoggingMiddleware)

# Include routers
app.include_router(auth_router)
app.include_router(loan_router)
app.include_router(document_router)
app.include_router(model_router)
app.include_router(reports_router)

@app.get("/")
async def root():
    return {"message": "Auth System API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

# Add this to debug available routes
@app.get("/debug/routes")
async def debug_routes():
    """Debug endpoint to see all available routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, 'name', 'N/A')
            })
    return {"routes": routes}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)