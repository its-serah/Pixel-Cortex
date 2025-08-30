# Pixel Cortex - Local IT Support Agent with XAI

A local-first IT Support ticketing system with Explainable AI (XAI) that provides transparent, policy-grounded decisions for ticket triage and resolution planning.

## Features

### Core Functionality
- **Ticket Management**: Create, assign, prioritize, update status, and close tickets
- **Automated Triage**: Rule-based classification with confidence scoring
- **Policy-Grounded Planning**: Action plans derived from official IT policies
- **Explainable AI**: Complete transparency in decision-making process

### XAI Components
- **Reasoning Trace**: Step-by-step decision process
- **Policy Citations**: Direct references to policy chunks
- **Confidence Scoring**: Quantified certainty in recommendations
- **Alternative Options**: Other approaches considered
- **Counterfactuals**: "What if" scenarios
- **Performance Telemetry**: Latency and processing metrics

### Security & Compliance
- **Append-Only Audit Log**: Hash-chained tamper-evident logging
- **PII Redaction**: Automatic redaction of sensitive information
- **RBAC**: Role-based access control (Admin, Agent, User)
- **Deterministic Processing**: Same input always produces same output

## Architecture

### Backend (FastAPI + PostgreSQL)
- **API Gateway**: JWT authentication with RBAC
- **Ticket Service**: CRUD operations with event tracking
- **Triage Service**: Rule-based categorization (regex/keyword matching)
- **Policy Retriever**: Hybrid BM25 + TF-IDF search over chunked documents
- **Planner Service**: Maps categories to policy-grounded action checklists
- **XAI Service**: Assembles comprehensive explanation objects
- **Audit Service**: Hash-chained logging for tamper detection

### Frontend (React + Tailwind CSS)
- **Ticket List View**: Overview of all tickets with filtering
- **Ticket Detail View**: Complete ticket information and timeline
- **XAI Explanation Panel**: Interactive exploration of AI decisions
- **Dashboard**: Statistics and recent activity

## Quick Start

### Lightweight RAG+KG (Local, no Docker)

This mode runs the optimized backend and the React frontend with the new RAG + Knowledge Graph and AI Workbench. It uses a fast local LLM (qwen2.5:0.5b via Ollama) and TF‑IDF for retrieval.

1) Install Ollama and pull the model

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a small, fast model
ollama pull qwen2.5:0.5b
```

2) Start the backend (lightweight demo)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate

# Minimal dependencies for the demo backend
pip install fastapi uvicorn[standard] scikit-learn networkx joblib python-jose[cryptography] passlib[bcrypt]

# Optional: audio processing (for /api/audio/transcribe)
pip install openai-whisper librosa soundfile SpeechRecognition pydub webrtcvad

# Run the simplified backend with RAG+KG + AI endpoints
uvicorn main_simple:app --host 0.0.0.0 --port 8000 --reload
```

3) Start the frontend

```bash
cd frontend
npm install
npm start
```

4) Use the app
- Sign in at http://localhost:3000 (demo users below)
- Navigate to "AI Workbench"
  - Click "Index Policies" (indexes ./policies or POLICIES_DIR)
  - Try RAG Search, KG Query, and LLM Chat (RAG‑augmented)

Logs are written to backend/logs/app.jsonl (one JSON line per request).

Demo accounts:
- Admin: admin / admin123
- Agent: agent1 / agent123
- User: user1 / user123

### Prerequisites
- Docker and Docker Compose
- Git

### Local Development Setup

1. **Clone and setup**:
```bash
git clone https://github.com/its-serah/Pixel-Cortex.git
cd Pixel-Cortex
cp .env.example .env
```

2. **Start services**:
```bash
make up
```

3. **Access the application**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Demo Accounts
- **Admin**: admin / admin123
- **Agent**: agent1 / agent123  
- **User**: user1 / user123

## API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user info
- `POST /api/auth/verify-token` - Verify JWT token

### Tickets
- `GET /api/tickets` - List tickets (with filtering)
- `POST /api/tickets` - Create new ticket (auto-triage if category not specified)
- `GET /api/tickets/{id}` - Get ticket details
- `PUT /api/tickets/{id}` - Update ticket
- `GET /api/tickets/{id}/explanation` - Get XAI explanation for ticket processing

### Triage & XAI
- `POST /api/triage/analyze` - Analyze ticket for category/priority with explanation
- `POST /api/triage/suggest-plan` - Generate action plan with full XAI explanation
- `GET /api/triage/categories` - Get available categories
- `GET /api/triage/priorities` - Get available priorities

### Policies

### RAG + KG (API)
- `POST /api/rag/index` – Index policies and build KG (body: { policies_dir?: string })
- `POST /api/rag/search` – Retrieve top chunks + LLM summary (body: { query: string, k?: number })
- `POST /api/kg/build` – Rebuild KG from current chunks
- `POST /api/kg/query` – Graph traversal from concepts (body: { concepts: string[], max_hops?: number })

### LLM
- `POST /api/llm/chat` – Chat with retrieval‑augmentation
  - Body: { message: string, conversation_history?: Message[], augment?: boolean (default true), k?: number (default 3) }
- `GET /api/policies/documents` - List policy documents
- `POST /api/policies/search` - Search policies using hybrid retrieval
- `POST /api/policies/reindex` - Reindex policy documents (admin only)
- `GET /api/policies/stats` - Get indexing statistics

### Users (Admin only)
- `GET /api/users` - List users
- `POST /api/users` - Create user
- `PUT /api/users/{id}` - Update user
- `DELETE /api/users/{id}` - Delete user

## XAI Explanation Object Schema

Every AI decision includes a comprehensive explanation:

```json
{
  "answer": "Human-readable summary of the decision",
  "decision": "Structured decision string (category=hardware, priority=high)",
  "confidence": 0.85,
  "reasoning_trace": [
    {
      "step": 1,
      "action": "text_analysis", 
      "rationale": "Analyzed input for keywords and patterns",
      "confidence": 0.9,
      "policy_refs": []
    }
  ],
  "policy_citations": [
    {
      "document_id": 1,
      "document_title": "Hardware Policy",
      "chunk_id": 5,
      "chunk_content": "When computer won't turn on, check power connections...",
      "relevance_score": 0.87
    }
  ],
  "missing_info": ["Specific error message", "Device model"],
  "alternatives_considered": [
    {
      "option": "Manual categorization",
      "pros": ["100% accuracy", "Human oversight"],
      "cons": ["Slower", "Requires expertise"],
      "confidence": 0.95
    }
  ],
  "counterfactuals": [
    {
      "condition": "If this were a security issue",
      "outcome": "Immediate isolation would be required",
      "likelihood": 0.1
    }
  ],
  "telemetry": {
    "latency_ms": 150,
    "retrieval_k": 5,
    "triage_time_ms": 100,
    "planning_time_ms": 50,
    "total_chunks_considered": 8
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "model_version": "1.0.0"
}
```

## Development Commands

```bash
# Build all services
make build

# Start services
make up

# View logs
make logs

# Stop services
make down

# Run backend tests
make test

# Clean Docker resources
make clean

# Install dependencies locally
make install-backend
make install-frontend

# Run services in development mode
make dev-backend
make dev-frontend
```

## Local Development (without Docker)

Note: For the fastest path, use the Lightweight RAG+KG steps above. The following section describes the original full stack setup.

1. **Backend setup**:
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

2. **Frontend setup**:
```bash
cd frontend
npm install
npm start
```

3. **Database setup**:
```bash
# Install PostgreSQL locally or use Docker
docker run -d --name pixel-postgres -e POSTGRES_DB=pixel_cortex -e POSTGRES_USER=pixel_user -e POSTGRES_PASSWORD=pixel_pass -p 5432:5432 postgres:15
```

## Policy Management

The system supports multiple policy document formats:

### Supported Formats
- **Markdown (.md)**: Rendered and chunked for search
- **PDF (.pdf)**: Text extracted and chunked
- **Text (.txt)**: Direct processing

### Adding Policies
1. Place policy documents in the `policies/` directory (or set POLICIES_DIR in .env)
2. In Lightweight mode, build index with: `POST /api/rag/index` (AI Workbench UI provides a button)
3. In the original stack, you can reindex via `POST /api/policies/reindex` (if enabled)

### Sample Policies Included
- `hardware_policy.md`: Hardware troubleshooting procedures
- `network_security_policy.md`: Network and security incident response
- `software_policy.md`: Software installation and support procedures

## Testing

### Backend Tests
```bash
cd backend
python -m pytest tests/ -v
```

### Test Categories
- **XAI Schema Validation**: Ensures explanation objects conform to schema
- **Policy Grounding**: Verifies citations are faithful to source policies
- **Determinism Tests**: Confirms same input produces same output
- **Audit Chain Validation**: Verifies tamper-evident logging integrity

### Sample Test Commands
```bash
# Test XAI schema validation
python -m pytest tests/test_xai_schema.py -v

# Test policy grounding
python -m pytest tests/test_policy_grounding.py -v

# Test determinism
python -m pytest tests/test_determinism.py -v
```

## Core Design Principles

### Deterministic
- Same input always produces same output and citations
- Deterministic ordering of search results and processing steps
- Reproducible explanations for audit and compliance

### Faithful
- Every claim tied to specific policy chunk citations
- No hallucination - all recommendations grounded in policies
- Explicit acknowledgment when policy guidance is insufficient

### Private
- PII automatically redacted from all logs and explanations
- Local-first architecture - no external API calls
- User data remains within organization boundaries

### Tamper-Evident
- Append-only hash-chained audit log
- Cryptographic verification of log integrity
- Detection of missing or modified entries

## Production Deployment Notes

### Security Hardening
1. Change default passwords in `.env`
2. Use strong JWT secret keys
3. Enable HTTPS with proper certificates
4. Configure firewall rules
5. Regular security updates

### Scaling Considerations
- PostgreSQL can be configured for high availability
- Backend can be horizontally scaled behind load balancer
- Policy indexing can be moved to background workers
- Add Redis for caching frequent queries

### Monitoring
- Health check endpoints available at `/health`
- Comprehensive telemetry in every XAI explanation
- Audit log provides complete activity trail
- Performance metrics tracked per operation

## Troubleshooting

### Common Issues

**Backend won't start**:
- Check PostgreSQL connection
- Verify environment variables in `.env`
- Ensure required Python packages are installed

**Frontend build errors**:
- Run `npm install` to ensure dependencies are current
- Check Node.js version (requires 16+)
- Verify API_URL is correctly configured

**Policy indexing fails**:
- Check `policies/` directory exists and contains supported files
- Verify file permissions allow reading
- Check backend logs for specific error messages

**Tests failing**:
- Ensure test database is clean: `rm test.db`
- Install test dependencies: `pip install -r requirements.txt`
- Run tests individually to isolate issues

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request with clear description

For questions or support, please open an issue in the GitHub repository.
