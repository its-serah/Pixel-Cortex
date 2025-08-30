.PHONY: help build up down logs test clean install-backend install-frontend kg-build kg-stats kg-list kg-visualize kg-export kg-test kg-coverage

help:
	@echo "Available commands:"
	@echo "  build           - Build all Docker images"
	@echo "  up              - Start all services"
	@echo "  down            - Stop all services"
	@echo "  logs            - Show logs from all services"
	@echo "  test            - Run backend tests"
	@echo "  clean           - Clean Docker resources"
	@echo "  install-backend - Install backend dependencies locally"
	@echo "  install-frontend- Install frontend dependencies locally"
	@echo ""
	@echo "Knowledge Graph commands:"
	@echo "  kg-build        - Build knowledge graph from policies"
	@echo "  kg-stats        - Show knowledge graph statistics"
	@echo "  kg-list         - List all concepts in the graph"
	@echo "  kg-visualize    - Create graph visualization"
	@echo "  kg-export       - Export graph to JSON"
	@echo "  kg-test         - Test KG-Enhanced RAG with sample query"
	@echo "  kg-coverage     - Analyze policy coverage in graph"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

test:
	cd backend && python -m pytest tests/ -v

clean:
	docker-compose down -v
	docker system prune -f

install-backend:
	cd backend && pip install -r requirements.txt

install-frontend:
	cd frontend && npm install

dev-backend:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm start

# Knowledge Graph Management
kg-build:
	cd backend && python scripts/manage_kg.py rebuild

kg-stats:
	cd backend && python scripts/manage_kg.py stats

kg-list:
	cd backend && python scripts/manage_kg.py list-concepts -v

kg-visualize:
	cd backend && python scripts/manage_kg.py visualize -o ../docs/knowledge_graph.png

kg-export:
	cd backend && python scripts/manage_kg.py export -f json -o ../docs/knowledge_graph.json

kg-test:
	cd backend && python scripts/manage_kg.py search -q "VPN connection issues with MFA"

kg-coverage:
	cd backend && python scripts/manage_kg.py analyze-coverage
