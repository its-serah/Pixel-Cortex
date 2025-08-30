# Local IT Support Agent with Explainable AI - Decision System

## Overview

This is the **complete implementation** of the Local IT Support Agent with Explainable AI (XAI) that you specified. The system now includes the final **Allowed / Denied / Needs Approval** decision logic based strictly on official policies, with all AI reasoning stored only in tamper-evident audit logs (hidden from users).

## Key Features ✅

### Core Decision System
- **Policy-Grounded Decisions**: Every decision (Allowed/Denied/Needs Approval) is based on official policy documents
- **Automatic Triage**: AI categorizes tickets and assigns priority
- **Decision Rules Engine**: Regex-based patterns for security violations, cost thresholds, and approval requirements
- **Explainable AI**: Full structured reasoning trace with citations, alternatives, and counterfactuals
- **Tamper-Evident Logging**: All decisions and explanations stored in hash-chained audit logs

### User Experience
- **Clean Decision Summaries**: Users see only final decision with friendly explanations
- **Hidden AI Reasoning**: Complex explanations stored securely in logs, not exposed in UI
- **Role-Based Access**: Different views for admins, agents, and regular users
- **React Frontend**: Modern UI with Tailwind CSS styling

### Security & Compliance
- **Audit Trail**: Append-only, cryptographically secured logs
- **PII Redaction**: Automatic removal of sensitive information from logs
- **Policy Citations**: Every decision references specific policy sections
- **Deterministic Output**: Consistent results for identical inputs

## System Architecture

```
Ticket → [Triage] → [Decision Engine] → [Planning] → [Audit Logging]
           ↓            ↓                ↓             ↓
       Category     Allowed/Denied   Action Plan   Tamper-Evident
       Priority     /Needs Approval               Secure Storage
```

### Decision Flow

1. **Triage**: AI analyzes title/description → category + priority
2. **Decision Engine**: Policy-grounded decision logic:
   - **DENIED**: Security violations, unauthorized access, policy breaches
   - **NEEDS_APPROVAL**: High cost, critical systems, management oversight required  
   - **ALLOWED**: Standard procedures, routine maintenance, approved requests
3. **Planning**: Generate resolution steps based on category and decision
4. **XAI Builder**: Merge all explanations into comprehensive reasoning trace
5. **Audit Logging**: Store full explanation with PII redaction and hash chaining
6. **User Response**: Return only clean decision summary (no internal reasoning)

## Decision Rules

### Automatic Denial
- Install unauthorized software
- Bypass security controls
- Personal/gaming software requests
- Disable firewall or security systems
- External access without authorization

### Requires Approval
- Hardware purchases over cost threshold
- Critical system modifications
- Production server changes
- Data recovery operations
- Major system changes

### Auto-Approved
- Password resets for standard users
- Software updates (approved list)
- Routine maintenance
- Minor cosmetic issues
- Standard procedure requests

## API Endpoints

### Core Decision Processing
```http
POST /api/tickets/{ticket_id}/process
```
**Returns**: Clean decision summary only (no AI reasoning exposed)
```json
{
  "ticket_id": 123,
  "decision": "allowed",
  "category": "access",
  "priority": "medium", 
  "summary": "✅ Access request (Medium priority) has been approved and will proceed according to standard procedures.",
  "confidence": 0.87,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Audit Trail (Admin Only)
```http
GET /api/audit/entries/{ticket_id}
```
**Returns**: Full reasoning trace with policy citations (secured)

## Testing the Decision System

Run the decision service tests:

```bash
cd backend
python -m pytest tests/test_decision_service.py -v
```

Test scenarios include:
- Security violation denials
- High-cost approval requirements  
- Standard request approvals
- Policy citation grounding
- Confidence calculations
- Missing information identification

## Example Decision Flow

### Input Ticket
```
Title: "Install Adobe Photoshop"
Description: "Need Photoshop for marketing materials, budget approved by manager"
```

### AI Processing (Hidden from User)
1. **Triage**: Category=SOFTWARE, Priority=MEDIUM
2. **Policy Retrieval**: Finds relevant software policy chunks
3. **Decision Logic**: Checks approval patterns, cost requirements
4. **Decision**: NEEDS_APPROVAL (requires manager authorization for paid software)
5. **Planning**: Generates procurement and installation steps
6. **Explanation**: Full reasoning with policy citations stored in audit logs

### User Sees Only
```json
{
  "decision": "needs_approval",
  "summary": "⏳ Software request (Medium priority) requires additional approval before proceeding. Management review has been requested.",
  "confidence": 0.91
}
```

### Audit Log Contains (Admin Only)
- Complete reasoning trace (15 steps)
- Policy citations (3 relevant chunks) 
- Alternatives considered (2 options)
- Counterfactual scenarios (3 scenarios)
- Confidence calculations
- Processing telemetry
- PII-redacted content

## Quick Start

1. **Setup Environment**:
```bash
cd backend
pip install -r requirements.txt
```

2. **Initialize Database**:
```bash
python scripts/seed_database.py
```

3. **Index Policies**:
```bash
python -c "from app.services.policy_indexer import PolicyIndexer; PolicyIndexer().index_all_policies('policies/')"
```

4. **Start Backend**:
```bash
uvicorn app.main:app --reload
```

5. **Start Frontend**:
```bash
cd frontend
npm install && npm start
```

6. **Test Decision System**:
```bash
# Create test ticket
curl -X POST http://localhost:8000/api/tickets/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{"title": "Install gaming software", "description": "Need Steam for personal use"}'

# Process ticket for decision  
curl -X POST http://localhost:8000/api/tickets/1/process \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Expected: {"decision": "denied", "summary": "❌ Software request denied..."}
```

## Architecture Highlights

### Explainable AI Framework
- **Structured Reasoning**: Every decision includes step-by-step rationale
- **Policy Grounding**: Citations link decisions to specific policy sections
- **Alternative Analysis**: "What if" scenarios for different approaches
- **Confidence Scoring**: Probabilistic assessment of decision quality

### Security-First Design
- **Zero Trust**: All decisions logged and auditable
- **PII Protection**: Sensitive data automatically redacted
- **Tamper Evidence**: Cryptographic hash chains prevent log modification
- **Role Separation**: Users see decisions, admins see reasoning

### Local-First Architecture
- **No External APIs**: Fully self-contained system
- **Policy Documents**: Stored as local markdown/PDF files
- **Vector Search**: BM25 + TF-IDF hybrid retrieval
- **Deterministic**: Same input always produces same output

## File Structure

```
backend/
├── app/
│   ├── services/
│   │   ├── decision_service.py      # Core decision logic
│   │   ├── xai_builder_service.py   # Explanation composition  
│   │   ├── policy_retriever.py      # Policy search
│   │   ├── audit_service.py         # Tamper-evident logging
│   │   └── privacy_utils.py         # PII redaction
│   ├── routers/
│   │   └── tickets.py               # Decision API endpoints
│   └── models/
│       └── schemas.py               # Explanation objects
├── policies/                        # Policy documents
├── tests/
│   └── test_decision_service.py     # Decision system tests
└── requirements.txt

frontend/
├── src/
│   ├── components/
│   │   ├── TicketDetail.js         # Clean decision display
│   │   └── Layout.js               # Navigation
│   └── utils/
│       └── api.js                  # API integration
```

## Production Deployment

For production use:

1. **Environment Variables**: Set secure JWT secrets, database URLs
2. **Reverse Proxy**: Use nginx for HTTPS termination  
3. **Database**: PostgreSQL with proper backups
4. **Monitoring**: Track decision accuracy and audit log integrity
5. **Policy Updates**: Establish process for updating policy documents

## Compliance & Audit

The system supports compliance requirements through:

- **Full Audit Trail**: Every decision is logged with reasoning
- **Policy Traceability**: Decisions linked to specific policy sections  
- **Tamper Evidence**: Cryptographic verification of log integrity
- **PII Protection**: Automatic redaction of sensitive information
- **Deterministic Replay**: Ability to recreate historical decisions

---

## Summary

This Local IT Support Agent with Explainable AI provides:

✅ **Complete Decision System**: Allowed/Denied/Needs Approval based on policies  
✅ **Hidden AI Reasoning**: Full explanations stored securely, not shown to users  
✅ **Policy Grounding**: Every decision references official documentation  
✅ **Tamper-Evident Logging**: Cryptographically secured audit trail  
✅ **Clean User Experience**: Simple decision summaries with friendly messaging  
✅ **Role-Based Access**: Appropriate views for different user types  
✅ **Local Deployment**: No external dependencies or API calls  
✅ **Comprehensive Testing**: Full test coverage for decision logic  

The system now fully implements your detailed requirements with all AI reasoning hidden from users while maintaining complete transparency for audit and compliance purposes through secure logging.
