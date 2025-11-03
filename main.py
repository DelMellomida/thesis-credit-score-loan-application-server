from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.auth_routes import router as auth_router
from app.api.loan_routes import router as loan_router
from app.api.document_routes import router as document_router
from app.api.model_routes import router as model_router
from contextlib import asynccontextmanager
from app.database.connection import init_db
from app.core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(
    title="Credit Scoring and Loan Recommendation",
    description="Credit Scoring and Loan Recommendation",
    version="1.0.0",
    lifespan=lifespan  # Add this line to use the lifespan context manager
)

# Add CORS middleware
# Support comma-separated CLIENT_URL values (e.g. "http://localhost:3000,http://localhost:3001")
raw_origins = settings.CLIENT_URL or ""
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

# Always include both development ports
if not allowed_origins or any("localhost" in origin for origin in allowed_origins):
    allowed_origins = list(set(allowed_origins + ["http://localhost:3000", "http://localhost:3001"]))

print(f"DEBUG: CORS allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(loan_router)
app.include_router(document_router)
app.include_router(model_router)

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