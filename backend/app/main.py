from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from app.core.database import engine
from app.models import models
from app.routers import auth, tickets, users, triage, policies, kg_router
from app.api import audio_router
from app.api import performance
from app.api import search
from app.services.policy_indexer import PolicyIndexer
from app.services.performance_monitor import performance_monitor
from app.core.seed import seed_database
import os

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Seed database with initial data
try:
    seed_database()
except Exception as e:
    print(f"Seeding failed: {e}")

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

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(tickets.router, prefix="/api/tickets", tags=["Tickets"])
app.include_router(triage.router, prefix="/api/triage", tags=["Triage"])
app.include_router(policies.router, prefix="/api/policies", tags=["Policies"])
app.include_router(kg_router.router)
app.include_router(audio_router.router, prefix="/api/audio", tags=["Audio Processing"])
app.include_router(performance.router, prefix="/api/performance", tags=["Performance Monitoring"])
app.include_router(search.router, prefix="/api/search", tags=["Interactive Search"])

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Initialize policy indexer
    policies_dir = os.getenv("POLICIES_DIR", "./policies")
    if os.path.exists(policies_dir):
        indexer = PolicyIndexer()
        await indexer.index_policies_directory(policies_dir)
        print(f"Indexed policies from {policies_dir}")
    
    # Start performance monitoring
    performance_monitor.start_monitoring(interval_seconds=30)
    print("Performance monitoring started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup services on shutdown"""
    # Stop performance monitoring
    performance_monitor.stop_monitoring()
    print("Performance monitoring stopped")

@app.get("/")
async def root():
    return {"message": "Pixel Cortex - Local IT Support Agent with XAI"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "pixel-cortex-backend"}
