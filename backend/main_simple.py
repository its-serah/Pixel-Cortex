from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
import re

# Model selector for Ollama
# Set OLLAMA_MODEL env var to override (e.g., qwen2.5:0.5b or llama3.2:3b)
def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL") or os.getenv("MODEL") or "llama3.2:3b"

# Create simple app without complex imports
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

# Simple JSONL logger (logs/app.jsonl)
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs") if os.path.dirname(__file__) else "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("pixel_cortex")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(os.path.join(LOG_DIR, "app.jsonl"))
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = int((time.time() - start) * 1000)
    user = None
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user = payload.get("sub")
        except Exception:
            user = None
    try:
        logger.info(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
            "user": user
        }))
    except Exception:
        pass
    return response

@app.get("/")
async def root():
    return {"message": "Pixel Cortex - Local IT Support Agent with XAI"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "pixel-cortex-backend"}

@app.get("/api/llm/test")
async def test_llm():
    """Test LLM functionality"""
    try:
        import ollama
        response = ollama.chat(
            model=get_ollama_model(),
            messages=[
                {"role": "system", "content": "You are an IT support assistant. Be brief and helpful."},
                {"role": "user", "content": "What is a VPN?"}
            ]
        )
        return {
            "status": "success",
            "llm_response": response['message']['content'],
            "model": get_ollama_model()
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "message": "LLM not available"
        }


# Helper to strip CoT and keep only final answer for user
def _clean_llm_text(txt: str) -> (str, str):
    if not txt:
        return "", ""
    low = txt.lower()
    # Prefer explicit Final Answer marker
    if "final answer:" in low:
        idx = low.find("final answer:")
        cot = txt[:idx].strip()
        final = txt[idx+len("final answer:"):].strip()
        return final, cot
    # Generic Answer marker
    if re.search(r"\banswer:\s*", low):
        m = re.search(r"(?is)(.*?)\banswer:\s*(.*)$", txt)
        if m:
            cot = (m.group(1) or "").strip()
            final = (m.group(2) or "").strip()
            return final, cot
    # Reasoning sections
    if re.search(r"(?i)reasoning\s*:", txt):
        # Remove reasoning block
        final = re.sub(r"(?is)reasoning\s*:[\s\S]*", "", txt).strip()
        cotm = re.search(r"(?is)reasoning\s*:\s*([\s\S]*)", txt)
        cot = cotm.group(1).strip() if cotm else ""
        return final, cot
    # No detectable CoT markers; return as-is
    return txt.strip(), ""


@app.get("/api/audio/test")
async def test_audio():
    """Test audio processing capabilities"""
    try:
        import whisper
        import librosa
        import speech_recognition as sr
        
        return {
            "status": "success",
            "audio_capabilities": {
                "whisper": "available",
                "librosa": "available", 
                "speech_recognition": "available",
                "microphones": len(sr.Microphone.list_microphone_names())
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Audio processing not available"
        }

@app.get("/api/performance/test")
async def test_performance():
    """Test performance monitoring"""
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "success",
            "metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / 1e9, 2),
                "memory_total_gb": round(memory.total / 1e9, 2),
                "disk_percent": round((disk.used / disk.total) * 100, 1)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Performance monitoring not available"
        }

# === AUTHENTICATION SYSTEM ===

# Security setup
SECRET_KEY = "your-secret-key-here-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Demo users (in production, this would be from database with proper hashing)
DEMO_USERS = {
    "admin": {
        "id": 1,
        "username": "admin",
        "email": "admin@pixelcortex.com",
        "full_name": "System Administrator",
        "role": "admin",
        "password": "admin123"  # Plain text for demo
    },
    "agent1": {
        "id": 2,
        "username": "agent1",
        "email": "agent1@pixelcortex.com",
        "full_name": "IT Support Agent",
        "role": "agent",
        "password": "agent123"  # Plain text for demo
    },
    "user1": {
        "id": 3,
        "username": "user1",
        "email": "user1@pixelcortex.com",
        "full_name": "Regular User",
        "role": "user",
        "password": "user123"  # Plain text for demo
    }
}
# Precompute demo password hashes to align with production practices (while preserving plaintext fallback for tests)
try:
    for _u in DEMO_USERS.values():
        _u['password_hash'] = get_password_hash(_u['password'])
except Exception:
    pass

class LoginRequest(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    role: str

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str):
    user = DEMO_USERS.get(username)
    if not user:
        return False
    # Prefer hashed verification if available
    try:
        ph = user.get('password_hash')
        if ph and verify_password(password, ph):
            return user
    except Exception:
        pass
    # Fallback to plaintext comparison for demo
    if password == user.get("password"):
        return user
    return False

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = DEMO_USERS.get(username)
    if user is None:
        raise credentials_exception
    return user

@app.post("/api/auth/login", response_model=Token)
async def login(login_data: LoginRequest):
    user = authenticate_user(login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/auth/me", response_model=User)
async def get_me(current_user: dict = Depends(get_current_user)):
    return User(
        id=current_user["id"],
        username=current_user["username"],
        email=current_user["email"],
        full_name=current_user["full_name"],
        role=current_user["role"]
    )

@app.get("/api/auth/verify-token")
async def verify_token(current_user: dict = Depends(get_current_user)):
    return {
        "valid": True,
        "user": {
            "id": current_user["id"],
            "username": current_user["username"],
            "email": current_user["email"],
            "full_name": current_user["full_name"],
            "role": current_user["role"]
        }
    }

# === TICKETS SYSTEM ===

# Demo tickets data
DEMO_TICKETS = {
    1: {
        "id": 1,
        "title": "VPN Connection Issues",
        "description": "Unable to connect to company VPN from home office. Getting timeout errors.",
        "status": "open",
        "priority": "high",
        "category": "network",
        "requester": "user1",
        "assigned_agent": "agent1",
        "created_at": "2024-08-30T10:00:00Z",
        "updated_at": "2024-08-30T10:00:00Z"
    },
    2: {
        "id": 2,
        "title": "Slow Computer Performance",
        "description": "Computer has been running very slowly for the past week. Takes long to start applications.",
        "status": "in_progress",
        "priority": "medium",
        "category": "hardware",
        "requester": "user1",
        "assigned_agent": "agent1",
        "created_at": "2024-08-29T14:30:00Z",
        "updated_at": "2024-08-30T09:15:00Z"
    },
    3: {
        "id": 3,
        "title": "Password Reset Request",
        "description": "Need to reset my password for the accounting system. Cannot remember current password.",
        "status": "resolved",
        "priority": "low",
        "category": "access",
        "requester": "user1",
        "assigned_agent": "admin",
        "created_at": "2024-08-28T11:20:00Z",
        "updated_at": "2024-08-28T16:45:00Z"
    }
}

class TicketCreate(BaseModel):
    title: str
    description: str
    category: str
    priority: str = "medium"

class TicketUpdate(BaseModel):
    title: str = None
    description: str = None
    status: str = None
    priority: str = None
    assigned_agent: str = None

@app.get("/api/tickets")
async def get_tickets(current_user: dict = Depends(get_current_user)):
    """Get all tickets (filtered by user role)"""
    tickets = list(DEMO_TICKETS.values())
    
    # Filter tickets based on user role
    if current_user["role"] == "user":
        # Users only see their own tickets
        tickets = [t for t in tickets if t["requester"] == current_user["username"]]
    
    return {"tickets": tickets, "total": len(tickets)}

@app.post("/api/tickets")
async def create_ticket(ticket_data: TicketCreate, current_user: dict = Depends(get_current_user)):
    """Create a new support ticket"""
    new_id = max(DEMO_TICKETS.keys()) + 1 if DEMO_TICKETS else 1
    
    new_ticket = {
        "id": new_id,
        "title": ticket_data.title,
        "description": ticket_data.description,
        "status": "open",
        "priority": ticket_data.priority,
        "category": ticket_data.category,
        "requester": current_user["username"],
        "assigned_agent": None,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    DEMO_TICKETS[new_id] = new_ticket
    
    return {"message": "Ticket created successfully", "ticket": new_ticket}

@app.get("/api/tickets/{ticket_id}")
async def get_ticket(ticket_id: int, current_user: dict = Depends(get_current_user)):
    """Get specific ticket by ID"""
    ticket = DEMO_TICKETS.get(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Check permissions
    if (current_user["role"] == "user" and 
        ticket["requester"] != current_user["username"]):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return ticket

class TicketStatusUpdate(BaseModel):
    status: str
    resolution_code: Optional[str] = None
    policy_refs: Optional[List[str]] = None

ALLOWED_STATUSES = ["open","in_progress","resolved","closed"]
ALLOWED_TRANSITIONS = {
    "open": {"in_progress","resolved","closed"},
    "in_progress": {"resolved","closed"},
    "resolved": {"closed"},
    "closed": set()
}

@app.patch("/api/tickets/{ticket_id}/status")
async def update_ticket_status(ticket_id: int, body: TicketStatusUpdate, current_user: dict = Depends(get_current_user)):
    ticket = DEMO_TICKETS.get(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    new_status = body.status.lower()
    if new_status not in ALLOWED_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid status")
    old = ticket["status"].lower()
    if new_status not in ALLOWED_TRANSITIONS.get(old, set()):
        raise HTTPException(status_code=400, detail=f"Invalid transition {old} -> {new_status}")
    ticket["status"] = new_status
    ticket["updated_at"] = datetime.utcnow().isoformat()
    # Log transition
    try:
        logger.info(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "type": "ticket_status_change",
            "ticket_id": ticket_id,
            "from": old,
            "to": new_status,
            "by": current_user.get("username"),
            "resolution_code": body.resolution_code,
            "policy_refs": body.policy_refs or []
        }))
    except Exception:
        pass
    return {"message": "status updated", "ticket": ticket}

# === SEARCH SYSTEM ===

class SearchRequest(BaseModel):
    query: str
    search_types: list = ["unified"]
    limit: int = 5

@app.post("/api/search/search")
async def search(search_data: SearchRequest, current_user: dict = Depends(get_current_user)):
    """Unified search across tickets, policies, and knowledge base"""
    query = search_data.query.lower()
    results = []
    
    # Search tickets
    for ticket in DEMO_TICKETS.values():
        if (query in ticket["title"].lower() or 
            query in ticket["description"].lower()):
            results.append({
                "type": "ticket",
                "id": ticket["id"],
                "title": ticket["title"],
                "description": ticket["description"][:200] + "...",
                "relevance_score": 0.8,
                "metadata": {
                    "status": ticket["status"],
                    "priority": ticket["priority"],
                    "category": ticket["category"]
                }
            })
    
    # Search knowledge base (simulated)
    knowledge_items = [
        {
            "type": "knowledge",
            "id": "kb_1",
            "title": "VPN Setup Guide",
            "description": "Complete guide for setting up VPN connections on Windows and Mac systems...",
            "relevance_score": 0.9,
            "metadata": {"category": "network", "difficulty": "intermediate"}
        },
        {
            "type": "knowledge",
            "id": "kb_2", 
            "title": "Password Security Best Practices",
            "description": "Guidelines for creating strong passwords and managing credentials securely...",
            "relevance_score": 0.7,
            "metadata": {"category": "security", "difficulty": "basic"}
        }
    ]
    
    for item in knowledge_items:
        if (query in item["title"].lower() or 
            query in item["description"].lower()):
            results.append(item)
    
    # Sort by relevance
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    results = results[:search_data.limit]
    
    # Generate AI summary of search results
    try:
        import ollama
        summary_prompt = f"Summarize these search results for the query '{search_data.query}': {str(results[:3])}"
        summary_response = ollama.chat(
            model=get_ollama_model(),
            messages=[
                {"role": "system", "content": "You are an IT support assistant. Provide brief summaries."},
                {"role": "user", "content": summary_prompt}
            ]
        )
        summary = summary_response['message']['content']
    except:
        summary = f"Found {len(results)} results related to '{search_data.query}'"
    
    return {
        "results": results,
        "total": len(results),
        "summary": summary,
        "query": search_data.query
    }

# === AUDIO PROCESSING ===

class AudioTranscribeRequest(BaseModel):
    audio_data: str  # base64 encoded audio
    format: str = "wav"

@app.post("/api/audio/transcribe")
async def transcribe_audio(audio_data: AudioTranscribeRequest, current_user: dict = Depends(get_current_user)):
    """Transcribe audio using Whisper"""
    try:
        import whisper
        import base64
        import io
        import tempfile
        import os
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data.audio_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{audio_data.format}", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        # Load Whisper model (using tiny for speed)
        model = whisper.load_model("tiny")
        
        # Transcribe
        result = model.transcribe(temp_file_path)
        
        # Clean up
        os.unlink(temp_file_path)
        
        return {
            "status": "success",
            "transcription": result["text"],
            "language": result["language"],
            "confidence": "high" if len(result["text"]) > 10 else "medium"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Audio transcription failed"
        }

@app.get("/api/audio/microphones")
async def get_microphones(current_user: dict = Depends(get_current_user)):
    """Get available microphones"""
    try:
        import speech_recognition as sr
        microphones = sr.Microphone.list_microphone_names()
        return {
            "status": "success",
            "microphones": microphones,
            "count": len(microphones)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "microphones": []
        }

# === PERFORMANCE MONITORING ===

@app.get("/api/performance/metrics")
async def get_performance_metrics(current_user: dict = Depends(get_current_user)):
    """Get current system performance metrics"""
    try:
        import psutil
        import time
        
        # Get comprehensive system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get process info
        process_count = len(psutil.pids())
        
        # Calculate health score (0-100)
        health_score = (
            (100 - cpu_percent) * 0.3 +
            (100 - memory.percent) * 0.4 +
            (100 - (disk.used / disk.total * 100)) * 0.3
        )
        
        return {
            "metrics": {
                "cpu": {
                    "percent": round(cpu_percent, 1),
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "percent": round(memory.percent, 1),
                    "used_gb": round(memory.used / 1e9, 2),
                    "total_gb": round(memory.total / 1e9, 2),
                    "available_gb": round(memory.available / 1e9, 2)
                },
                "disk": {
                    "percent": round((disk.used / disk.total) * 100, 1),
                    "used_gb": round(disk.used / 1e9, 2),
                    "total_gb": round(disk.total / 1e9, 2),
                    "free_gb": round(disk.free / 1e9, 2)
                },
                "processes": process_count,
                "timestamp": datetime.utcnow().isoformat()
            },
            "health_score": round(health_score, 1),
            "status": "healthy" if health_score > 70 else "warning" if health_score > 50 else "critical"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Performance metrics not available"
        }

class MonitoringRequest(BaseModel):
    interval_seconds: int = 30

@app.post("/api/performance/monitoring/start")
async def start_monitoring(monitoring_data: MonitoringRequest, current_user: dict = Depends(get_current_user)):
    """Start performance monitoring"""
    # In a real implementation, this would start a background task
    return {
        "status": "started",
        "interval_seconds": monitoring_data.interval_seconds,
        "message": f"Performance monitoring started with {monitoring_data.interval_seconds}s interval"
    }

@app.post("/api/performance/monitoring/stop")
async def stop_monitoring(current_user: dict = Depends(get_current_user)):
    """Stop performance monitoring"""
    return {
        "status": "stopped",
        "message": "Performance monitoring stopped"
    }

# === TRIAGE SYSTEM ===

class TriageRequest(BaseModel):
    title: str
    description: str
    user_role: str = "user"

@app.post("/api/triage/analyze")
async def analyze_ticket(triage_data: TriageRequest, current_user: dict = Depends(get_current_user)):
    """AI-powered ticket triage and categorization"""
    try:
        import ollama
        
        triage_prompt = f"""Analyze this IT support request and provide triage information:

Title: {triage_data.title}
Description: {triage_data.description}
User Role: {triage_data.user_role}

Provide your analysis in this format:
Category: [network|hardware|software|security|access|other]
Priority: [low|medium|high|critical]
Estimated Resolution Time: [minutes|hours|days]
Key Issues: [list main problems]
Recommended Actions: [immediate steps to take]
"""
        
        response = ollama.chat(
            model=get_ollama_model(),
            messages=[
                {"role": "system", "content": "You are an expert IT support triage specialist. Analyze requests quickly and accurately."},
                {"role": "user", "content": triage_prompt}
            ],
            options={"num_predict": int(os.getenv("LLM_TRIAGE_NUM_PREDICT", "200"))}
        )
        
        analysis = response['message']['content']
        
        # Extract structured data (simplified)
        category = "other"
        priority = "medium"
        
        analysis_lower = analysis.lower()
        if "network" in analysis_lower: category = "network"
        elif "hardware" in analysis_lower: category = "hardware"
        elif "software" in analysis_lower: category = "software"
        elif "security" in analysis_lower: category = "security"
        elif "access" in analysis_lower or "password" in analysis_lower: category = "access"
        
        if "critical" in analysis_lower: priority = "critical"
        elif "high" in analysis_lower: priority = "high"
        elif "low" in analysis_lower: priority = "low"
        
        return {
            "status": "success",
            "triage_result": {
                "category": category,
                "priority": priority,
                "confidence": 0.85,
                "analysis": analysis,
                "estimated_time": "hours",
                "requires_escalation": priority in ["high", "critical"]
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Triage analysis failed"
        }

# === KNOWLEDGE SYSTEM ===

@app.get("/api/knowledge/policies")
async def get_policies(current_user: dict = Depends(get_current_user)):
    """Get IT policies and procedures"""
    policies = [
        {
            "id": 1,
            "title": "VPN Access Policy",
            "description": "Guidelines for secure VPN usage and access management",
            "category": "network",
            "last_updated": "2024-08-15T00:00:00Z"
        },
        {
            "id": 2,
            "title": "Password Security Requirements",
            "description": "Company standards for password creation and management",
            "category": "security",
            "last_updated": "2024-08-10T00:00:00Z"
        },
        {
            "id": 3,
            "title": "Software Installation Procedures",
            "description": "Process for requesting and installing new software",
            "category": "software",
            "last_updated": "2024-07-20T00:00:00Z"
        }
    ]
    
    return {"policies": policies, "total": len(policies)}

@app.get("/api/knowledge/faq")
async def get_faq(current_user: dict = Depends(get_current_user)):
    """Get frequently asked questions"""
    faq = [
        {
            "id": 1,
            "question": "How do I connect to the company VPN?",
            "answer": "Use the Cisco AnyConnect client with your domain credentials. Contact IT if you need the server address.",
            "category": "network",
            "views": 156
        },
        {
            "id": 2,
            "question": "How do I reset my password?",
            "answer": "Use the self-service portal at portal.company.com or contact the IT helpdesk for assistance.",
            "category": "access",
            "views": 89
        },
        {
            "id": 3,
            "question": "My computer is running slowly, what should I do?",
            "answer": "Try restarting first, then check for Windows updates, run disk cleanup, and scan for malware.",
            "category": "hardware",
            "views": 234
        }
    ]
    
    return {"faq": faq, "total": len(faq)}

# === RAG + KG ENDPOINTS ===

from app.services.ragkg_service import RAGKG
_ragkg = RAGKG()

class RAGIndexRequest(BaseModel):
    policies_dir: Optional[str] = None

@app.post("/api/rag/index")
async def rag_index(body: RAGIndexRequest, current_user: dict = Depends(get_current_user)):
    policies_dir = body.policies_dir or os.getenv("POLICIES_DIR", os.path.join(os.path.dirname(__file__) or ".", "policies"))
    if not os.path.isabs(policies_dir):
        policies_dir = os.path.abspath(os.path.join(os.path.dirname(__file__) or ".", "..", policies_dir))
    res = _ragkg.index_policies(policies_dir)
    return {"status": "ok", **res, "dir": policies_dir}

class RAGSearchRequest(BaseModel):
    query: str
    k: int = 5

@app.post("/api/rag/search")
async def rag_search(body: RAGSearchRequest, current_user: dict = Depends(get_current_user)):
    results = _ragkg.search(body.query, k=body.k)
    # Summarize with qwen for a quick, short answer
    summary = ""
    try:
        import ollama
        ctx = "\n---\n".join([r["content"][:400] for r in results[:3]])
        prompt = f"Question: {body.query}\nContext:\n{ctx}\n\nAnswer concisely and technically."
        resp = ollama.chat(
            model=get_ollama_model(),
            messages=[{"role":"system","content":"You are an IT support assistant."},{"role":"user","content":prompt}],
            options={"num_predict": int(os.getenv("LLM_SUMMARY_NUM_PREDICT", "300"))}
        )
        summary = resp['message']['content']
    except Exception:
        summary = f"Retrieved {len(results)} policy chunks for '{body.query}'."
    return {"results": results, "summary": summary}

class KGQueryRequest(BaseModel):
    concepts: List[str]
    max_hops: int = 2

@app.post("/api/kg/query")
async def kg_query(body: KGQueryRequest, current_user: dict = Depends(get_current_user)):
    res = _ragkg.query_graph(body.concepts, max_hops=body.max_hops)
    return res

@app.post("/api/kg/build")
async def kg_build(current_user: dict = Depends(get_current_user)):
    res = _ragkg.build_kg()
    return {"status": "ok", **res}

# Enhance chat via RAG (optional): if top chunks exist, prepend as context
# Modify ChatMessage to support augment flag and k
from pydantic import Field

class ChatMessage(BaseModel):
    message: str
    conversation_history: list = Field(default_factory=list)
    augment: bool = True
    k: int = 3

@app.post("/api/llm/chat")
async def chat_with_llm(chat_data: ChatMessage, current_user: dict = Depends(get_current_user)):
    try:
        import ollama
        messages = [{
            "role": "system",
            "content": (
                "You are Pixel Cortex, an expert IT Support Assistant. "
                "Do NOT include your chain-of-thought or internal reasoning in responses. "
                "If you need to reason, do so silently and return only the final answer. "
                "Format strictly as: 'Answer: <final answer>'. "
                "Provide numbered steps only when necessary for actions, not your internal thoughts."
            )
        }]
        if chat_data.conversation_history:
            messages.extend(chat_data.conversation_history[-10:])
        messages.append({"role": "user", "content": chat_data.message})
# Build enumerated policy context to enable citations
        citations = []
        if chat_data.augment:
            top = _ragkg.search(chat_data.message, k=min(5, max(1, chat_data.k)))
            if top:
                ctx = []
                for i, t in enumerate(top):
                    ref = f"[{i+1}]"
                    citations.append({"reference": ref, "title": t["document_title"], "chunk_id": t["chunk_id"]})
                    ctx.append(f"{ref} {t['document_title']}: {t['content'][:600]}")
                messages.append({"role": "system", "content": "Relevant policy context:\n" + "\n\n".join(ctx)})
        
        # Require strict JSON output
        messages.append({
            "role": "system",
            "content": (
                "Return ONLY valid JSON with this exact schema: "
                "{\"decision\":\"allowed|denied|requires_approval\","
                "\"decision_reason\":\"string\","
                "\"checklist\":[\"step 1\",\"step 2\"],"
                "\"policy_citations\":[{\"title\":\"string\",\"reference\":\"[1]\"}],"
                "\"notes\":\"string of missing details or caveats\"}. "
                "No markdown, no extra prose."
            )
        })
        _t0 = time.time()
        resp = ollama.chat(
            model=get_ollama_model(),
            messages=messages,
            options={
                "num_predict": int(os.getenv("LLM_NUM_PREDICT", "400")),
                "temperature": float(os.getenv("LLM_TEMPERATURE", "0.4"))
            }
        )
        inference_ms = int((time.time() - _t0) * 1000)
        raw = resp['message']['content']

        # Sanitize to extract JSON
        txt = raw.strip()
        if not (txt.startswith("{") and txt.endswith("}")):
            # try to slice between first { and last }
            if "{" in txt and "}" in txt:
                txt = txt[txt.find("{"):txt.rfind("}")+1]
        structured = None
        try:
            structured = json.loads(txt)
        except Exception:
            structured = None
        
        # Normalize structured fields
        allowed = {"allowed","denied","requires_approval"}
        decision = None
        decision_reason = ""
        checklist = []
        policy_citations = []
        notes = ""
        if isinstance(structured, dict):
            decision = str(structured.get("decision","requires_approval")).lower()
            if decision not in allowed:
                decision = "requires_approval"
            decision_reason = str(structured.get("decision_reason",""))
            checklist = [str(s) for s in structured.get("checklist", []) if isinstance(s, str)]
            pc = structured.get("policy_citations", [])
            if isinstance(pc, list):
                for c in pc:
                    if isinstance(c, dict) and "title" in c and "reference" in c:
                        policy_citations.append({"title": str(c["title"]), "reference": str(c["reference"])})
            notes = str(structured.get("notes",""))
        else:
            decision = "requires_approval"
        
        # Resolve references to chunk ids (server-side mapping)
        ref_map = {c["reference"]: c for c in citations}
        resolved_citations = []
        for c in policy_citations:
            rc = ref_map.get(c["reference"]) or {"reference": c["reference"], "title": c["title"], "chunk_id": None}
            resolved_citations.append(rc)
        
        # Build user-facing response text
        lines = []
        lines.append(f"Decision: {decision}" + (f" — {decision_reason}" if decision_reason else ""))
        if checklist:
            lines.append("Checklist:")
            for i, step in enumerate(checklist, 1):
                lines.append(f"  {i}. {step}")
        if resolved_citations:
            lines.append("Citations:")
            for rc in resolved_citations:
                lines.append(f"  {rc['reference']} {rc['title']}")
        if notes:
            lines.append(f"Notes: {notes}")
        final_text = "\n".join(lines)
        
        # Log CoT if any
        final_text_clean, cot = _clean_llm_text(final_text)
        if cot:
            try:
                logger.info(json.dumps({
                    "ts": datetime.utcnow().isoformat(),
                    "type": "llm_chat_cot",
                    "user": current_user.get("username"),
                    "prompt": chat_data.message,
                    "cot": cot[:4000]
                }))
            except Exception:
                pass
        # Log structured decision
        try:
            logger.info(json.dumps({
                "ts": datetime.utcnow().isoformat(),
                "type": "llm_chat_structured",
                "user": current_user.get("username"),
                "prompt": chat_data.message,
                "decision": decision,
                "checklist_len": len(checklist),
                "citations": resolved_citations,
                "inference_ms": inference_ms
            }))
        except Exception:
            pass
        return {"status": "success", "response": final_text_clean, "model": get_ollama_model(), "structured": {"decision": decision, "decision_reason": decision_reason, "checklist": checklist, "policy_citations": policy_citations, "notes": notes, "citations_resolved": resolved_citations}, "inference_ms": inference_ms}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Legacy simple chat (now placed after auth to avoid import-time errors)
@app.post("/api/llm/chat_simple")
async def chat_with_llm_simple(chat_data: ChatMessage, current_user: dict = Depends(get_current_user)):
    try:
        import ollama
        messages = [{
            "role": "system",
            "content": (
                "You are Pixel Cortex, an expert IT Support Assistant. "
                "Do NOT include your chain-of-thought or internal reasoning in responses. "
                "If you need to reason, do so silently and return only the final answer. "
                "Format strictly as: 'Answer: <final answer>'. "
                "Provide numbered steps only when necessary for actions."
            )
        }]
        if chat_data.conversation_history:
            messages.extend(chat_data.conversation_history[-10:])
        messages.append({"role": "user", "content": chat_data.message})
        resp = ollama.chat(
            model=get_ollama_model(),
            messages=messages,
            options={
                "temperature": float(os.getenv("LLM_TEMPERATURE", "0.6")),
                "top_p": float(os.getenv("LLM_TOP_P", "0.9")),
                "num_predict": int(os.getenv("LLM_SIMPLE_NUM_PREDICT", "500"))
            }
        )
        raw = resp['message']['content']
        final_text, cot = _clean_llm_text(raw)
        if cot:
            try:
                logger.info(json.dumps({
                    "ts": datetime.utcnow().isoformat(),
                    "type": "llm_chat_cot",
                    "user": current_user.get("username"),
                    "prompt": chat_data.message,
                    "cot": cot[:4000]
                }))
            except Exception:
                pass
        return {"status": "success", "response": final_text, "model": get_ollama_model()}
    except Exception as e:
        return {"status": "error", "error": str(e), "message": "LLM chat not available"}

# === ANALYTICS ===

@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics(current_user: dict = Depends(get_current_user)):
    """Get dashboard analytics data"""
    total_tickets = len(DEMO_TICKETS)
    open_tickets = len([t for t in DEMO_TICKETS.values() if t["status"] == "open"])
    
    return {
        "ticket_stats": {
            "total": total_tickets,
            "open": open_tickets,
            "in_progress": 1,
            "resolved": total_tickets - open_tickets - 1
        },
        "category_breakdown": {
            "network": 1,
            "hardware": 1,
            "access": 1,
            "software": 0,
            "security": 0,
            "other": 0
        },
        "priority_breakdown": {
            "critical": 0,
            "high": 1,
            "medium": 1,
            "low": 1
        },
        "response_times": {
            "avg_first_response_hours": 2.5,
            "avg_resolution_hours": 24.0,
            "sla_compliance_percent": 95.0
        }
    }
