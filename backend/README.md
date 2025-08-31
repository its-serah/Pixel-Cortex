# Pixel Cortex - Intelligent IT Support System

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Pixel Cortex is an intelligent IT support system that combines Knowledge Graph-Enhanced RAG (Retrieval-Augmented Generation) with explainable AI to provide deterministic, auditable IT support automation. The system is designed to be lightweight, deployable on free tiers (like Render.com), and operates without requiring expensive GPU resources.

### Key Features

- **Intelligent Triage**: Automatically categorize and prioritize IT tickets using pattern matching and NLP
- **Knowledge Graph RAG**: Enhanced retrieval with concept relationships and semantic understanding
- **Hybrid Search**: Combines BM25 and TF-IDF for optimal document retrieval
- **Audio Processing**: CPU-based audio transcription using Vosk (no GPU required)
- **Performance Monitoring**: Real-time metrics and trend analysis
- **Audit Trail**: Complete explainability with reasoning traces and decision logs
- **Lightweight Deployment**: Runs on free tiers with < 500MB memory footprint

## Architecture

```
Pixel-Cortex/
├── backend/
│   ├── app/
│   │   ├── api/          # API endpoints (search, audio, KG, chat)
│   │   ├── core/         # Core functionality (auth, database, config)
│   │   ├── models/       # SQLAlchemy models and Pydantic schemas
│   │   ├── routers/      # FastAPI routers
│   │   └── services/     # Business logic services
│   ├── policies/         # Policy documents for RAG
│   ├── tests/           # Test suite
│   ├── run_basic.py     # Lightweight application runner
│   ├── requirements.txt # Minimal dependencies
│   └── render.yaml      # Render deployment configuration
└── frontend/            # React frontend (optional)
```

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/pixel-cortex.git
cd pixel-cortex/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export AUTH_DISABLED=true  # For demo mode
export DEMO_MODE=true

# Run the application
python run_basic.py

# Access the API
# Documentation: http://localhost:8000/docs
# Health check: http://localhost:8000/health
```

### Deploy to Render

1. Fork this repository
2. Connect your GitHub to Render
3. Deploy using the `render.yaml` configuration
4. Your app will be available at `https://your-app.onrender.com`

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite:///./pixel_cortex.db` |
| `SECRET_KEY` | JWT secret key | (required for production) |
| `AUTH_DISABLED` | Disable authentication | `false` |
| `DEMO_MODE` | Enable demo mode | `false` |
| `USE_VOSK` | Enable Vosk audio processing | `false` |

## API Documentation

### Core Endpoints

#### Triage Service
```bash
# Analyze and categorize a ticket
curl -X POST http://localhost:8000/api/triage/analyze \
  -H "Content-Type: application/json" \
  -d '{"title": "VPN not working", "description": "Cannot connect from home"}'
```

#### Performance Metrics
```bash
# Get system performance metrics
curl http://localhost:8000/api/performance/metrics
```

#### Search Suggestions
```bash
# Get search suggestions
curl "http://localhost:8000/api/search/suggestions?query=vpn"
```

### Response Example

```json
{
  "category": "network",
  "priority": "medium",
  "confidence": 0.8,
  "explanation": {
    "answer": "Ticket categorized as network with medium priority",
    "reasoning_trace": [
      {
        "step": 1,
        "action": "text_analysis",
        "rationale": "Identified VPN and connection keywords",
        "confidence": 0.9
      }
    ],
    "telemetry": {
      "latency_ms": 2,
      "triage_time_ms": 2
    }
  }
}
```

## Knowledge Graph RAG

The system uses a Knowledge Graph-Enhanced RAG approach:

1. **Concept Extraction**: Identifies IT concepts (VPN, MFA, etc.) from queries
2. **Graph Traversal**: Explores related concepts through relationships
3. **Enhanced Retrieval**: Combines semantic search with graph-based relevance
4. **Explainable Results**: Provides reasoning traces for all decisions

### Example Concept Graph
```
VPN --[requires]--> MFA
 |                   |
 |                   v
 +--[depends_on]--> Network Access
```

## Features in Detail

### Deterministic & Auditable
- All decisions include reasoning traces
- Reproducible results for compliance
- Complete audit logs with hashes

### Lightweight Deployment
- No GPU required
- Runs on free cloud tiers
- < 500MB total deployment size
- CPU-only inference

### Extensible Architecture
- Plugin-based service architecture
- Easy to add new analyzers
- Configurable reasoning pipelines

## Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_triage_service.py

# Run with coverage
pytest --cov=app --cov-report=html
```

## Performance

- **Triage Response**: < 50ms
- **Search Response**: < 100ms  
- **Audio Processing**: Real-time with Vosk
- **Memory Usage**: < 500MB
- **Deployment Size**: < 500MB total

## Security

- JWT-based authentication
- Role-based access control (Admin, Agent, User)
- Audit trail with cryptographic hashes
- Environment-based configuration
- SQL injection protection via SQLAlchemy

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FastAPI for the excellent web framework
- Vosk for lightweight speech recognition
- scikit-learn for text processing
- The open-source community

## Contact

- GitHub Issues: [Report a bug](https://github.com/yourusername/pixel-cortex/issues)
- Email: your-email@example.com

---

**Built for the IT Support Community**
