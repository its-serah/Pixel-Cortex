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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include basic routers (without heavy LLM dependencies)
try:
    from app.routers import auth, tickets, users, policies
    app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
    app.include_router(users.router, prefix="/api/users", tags=["Users"])
    app.include_router(tickets.router, prefix="/api/tickets", tags=["Tickets"])
    app.include_router(policies.router, prefix="/api/policies", tags=["Policies"])
    print("Basic routers loaded successfully")
except ImportError as e:
    print(f"Warning: Some routers failed to load: {e}")

# CPU Audio (Vosk) endpoints
try:
    from app.api import audio_cpu
    app.include_router(audio_cpu.router, prefix="/api/audio", tags=["Audio CPU"])
    print("Vosk CPU audio endpoints enabled")
except Exception as e:
    print(f"Audio CPU router not available: {e}")

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

@app.get("/")
async def root():
    return {"message": "Pixel Cortex - Basic Mode (No LLM/Audio)", "status": "running"}

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
