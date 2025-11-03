from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.auth_routes import router as auth_router
from app.api.loan_routes import router as loan_router
from app.api.document_routes import router as document_router
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.CLIENT_URL],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(loan_router)
app.include_router(document_router)

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