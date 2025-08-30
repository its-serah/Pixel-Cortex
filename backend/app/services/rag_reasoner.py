"""
Deterministic CPU RAG Reasoner
- BM25+TFIDF top-k retrieval with stable ordering
- Optional KG-lite boost using concept mentions and co-occurrence graph
- Template-based answer composition with reasoning trace and citations
"""

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from app.services.policy_retriever import PolicyRetriever
from app.services.kg_lite_service import kg_lite_service


@dataclass
class ReasonStep:
    step: int
    action: str
    rationale: str


class RAGReasoner:
    def __init__(self):
        self.retriever = PolicyRetriever()

    def _decide_search(self, query: str) -> bool:
        q = (query or "").strip().lower()
        if not q or len(q.split()) < 2:
            return False
        trivial = ["hello", "hi", "thanks", "ok", "hey"]
        if q in trivial:
            return False
        return True

    def _boost_with_kg(self, db: Session, query: str, citations: List[Any]) -> Tuple[List[Any], Dict[str, Any]]:
        concepts = kg_lite_service.find_concepts_in_text(query)
        if not concepts:
            return citations, {"kg_enabled": False, "concepts": []}
        # Map concepts to chunk ids
        concept_chunks = set()
        for cname in concepts:
            # fetch concept by name
            from app.models.models import KnowledgeGraphConcept
            c = db.query(KnowledgeGraphConcept).filter(KnowledgeGraphConcept.name == cname).first()
            if c and c.policy_chunks:
                concept_chunks.update(c.policy_chunks)
        # Apply a small boost to citations coming from those chunks
        boosted = []
        for c in citations:
            score = c.relevance_score
            if c.chunk_id in concept_chunks:
                score += 0.1  # small deterministic boost
            boosted.append((score, c))
        boosted.sort(key=lambda x: (-x[0], x[1].chunk_id))
        return [c for _, c in boosted], {"kg_enabled": True, "concepts": concepts, "boosted": True}

    def chat(self, db: Session, message: str, augment: bool = True, k: int = 5, include_explanation: bool = True) -> Dict[str, Any]:
        steps: List[ReasonStep] = []
        start = datetime.now()
        used_retrieval = False
        citations = []
        graph_meta: Dict[str, Any] = {}

        steps.append(ReasonStep(1, "query_analysis", f"Analyzed user query: '{message[:100]}'"))
        if augment and self._decide_search(message):
            used_retrieval = True
            steps.append(ReasonStep(2, "policy_retrieval", "Retrieving relevant policy chunks (BM25 + TF-IDF)"))
            citations = self.retriever.retrieve_relevant_chunks(message, k=k, db=db)
            # KG-lite boost
            citations, graph_meta = self._boost_with_kg(db, message, citations)
        else:
            steps.append(ReasonStep(2, "skip_retrieval", "Heuristic decided retrieval not necessary"))

        # Compose answer deterministically
        answer_lines = []
        if used_retrieval and citations:
            top = citations[0]
            answer_lines.append(f"Based on policy '{top.document_title}':")
            content = top.chunk_content.strip().replace("\n", " ")
            snippet = (content[:280] + "...") if len(content) > 280 else content
            answer_lines.append(snippet)
            # Checklist template (extract simple imperative-like sentences)
            from re import findall
            sentences = [s.strip() for s in top.chunk_content.split('.') if s.strip()]
            actions = []
            for s in sentences:
                if len(actions) >= 5:
                    break
                if any(s.lower().startswith(v) for v in ["verify", "check", "ensure", "reset", "update", "apply", "document", "test", "notify"]):
                    actions.append(s.capitalize())
            if actions:
                answer_lines.append("Checklist:")
                for i, a in enumerate(actions, 1):
                    answer_lines.append(f"{i}. {a}.")
        else:
            answer_lines.append("I’m ready to help. Please provide more details or ask a specific IT policy question.")

        # Decision hint (allowed/denied/needs_approval) heuristic
        decision = "needs_approval" if any(w in message.lower() for w in ["purchase", "expensive", "admin rights"]) else "allowed"
        steps.append(ReasonStep(3, "decision_composition", f"Composed decision hint: {decision}"))

        latency_ms = int((datetime.now() - start).total_seconds() * 1000)

        response = "\n".join(answer_lines)
        result: Dict[str, Any] = {"response": response}
        if include_explanation:
            result["explanation"] = {
                "reasoning_trace": [s.__dict__ for s in steps],
                "policy_citations": [
                    {
                        "document_id": c.document_id,
                        "document_title": c.document_title,
                        "chunk_id": c.chunk_id,
                        "relevance_score": c.relevance_score
                    } for c in (citations or [])
                ],
                "kg": graph_meta,
                "decision": decision,
                "telemetry": {"latency_ms": latency_ms, "used_retrieval": used_retrieval, "k": k}
            }
        return result

rag_reasoner = RAGReasoner()

