"""
Basic startup script for Pixel Cortex without heavy ML dependencies

This script allows running the core FastAPI application without loading
LLM and audio processing services that require large ML dependencies.
"""

import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Set environment variables
os.environ.setdefault("DATABASE_URL", "sqlite:///./pixel_cortex.db")
os.environ.setdefault("SECRET_KEY", "dev_secret_key_change_in_production")
os.environ.setdefault("ALGORITHM", "HS256")
os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
os.environ.setdefault("POLICIES_DIR", "./policies")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.database import engine
from app.models import models
from app.core.migrations import ensure_ticket_columns

# Ensure new columns exist before running
ensure_ticket_columns(engine)
# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Pixel Cortex - Local IT Support Agent",
    description="Local-first IT Support Agent with Explainable AI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include basic routers (without heavy LLM dependencies)
try:
    from app.routers import auth, tickets, users, policies, triage
    app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
    app.include_router(users.router, prefix="/api/users", tags=["Users"])
    app.include_router(tickets.router, prefix="/api/tickets", tags=["Tickets"])
    app.include_router(policies.router, prefix="/api/policies", tags=["Policies"])
    app.include_router(triage.router, prefix="/api/triage", tags=["Triage"])
    print("Core routers loaded successfully")
except ImportError as e:
    print(f"Warning: Some routers failed to load: {e}")

# Minimal routers only
# KG Lite endpoints (no spaCy/networkx)
try:
    from app.api import kg_lite
    app.include_router(kg_lite.router, prefix="/api/kg-lite", tags=["Knowledge Graph (Lite)"])
    print("KG Lite endpoints enabled")
except Exception as e:
    print(f"KG Lite router not available: {e}")

# LLM Chat (Ollama) endpoints
try:
    from app.api import llm_chat
    app.include_router(llm_chat.router, prefix="/api/llm", tags=["LLM Chat"])
    print("LLM chat endpoints enabled")
except Exception as e:
    print(f"LLM chat router not available: {e}")

# RAG API (index + search) for policies
try:
    from app.api import rag_api
    app.include_router(rag_api.router, prefix="/api/rag", tags=["RAG"])
    print("RAG API endpoints enabled")
except Exception as e:
    print(f"RAG API router not available: {e}")

# IT Support Agent (LLM+RAG)
try:
    from app.api import it_agent
    app.include_router(it_agent.router, prefix="/api/agent", tags=["IT Agent"])
    print("IT Support Agent enabled (Ollama + RAG)")
except Exception as e:
    print(f"IT Agent not available: {e}")

# Serve the web interface if static folder exists
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/")
async def root():
    # Serve the HTML interface if it exists
    html_file = static_path / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    return {"message": "Pixel Cortex - IT Support Agent with Local LLM", "status": "running"}

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: StarletteHTTPException):
    """Serve a styled 404 page for browsers; JSON for API clients."""
    try:
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            html_404 = static_path / "404.html"
            if html_404.exists():
                return FileResponse(html_404, status_code=404)
    except Exception:
        pass
    return JSONResponse({"detail": "Not Found"}, status_code=404)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "pixel-cortex-backend-basic", "mode": "basic"}

@app.get("/features")
async def get_features():
    return {
        "available_features": [
            "User authentication",
            "Ticket management", 
            "Policy management",
            "Basic API endpoints"
        ],
        "disabled_features": [
            "LLM processing (requires torch, transformers)",
            "Audio processing (requires whisper, librosa)",
            "Performance monitoring (requires psutil)",
            "Interactive search (requires sentence-transformers)"
        ],
        "note": "Install ML dependencies to enable full functionality"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
