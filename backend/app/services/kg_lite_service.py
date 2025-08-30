"""
KG Lite Service (no spaCy/networkx)
- Lexicon-based concept detection
- Co-occurrence relationships between concepts per policy chunk
- Simple BFS path via SQL
"""

from typing import List, Tuple, Dict, Any, Set
from collections import defaultdict, deque
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.models import KnowledgeGraphConcept, PolicyChunk

# Small curated lexicon
IT_CONCEPTS: Dict[str, Dict[str, Any]] = {
    "VPN": {"aliases": ["vpn", "virtual private network", "remote access vpn"], "type": "technology", "importance": 0.9},
    "MFA": {"aliases": ["mfa", "multi-factor authentication", "two-factor authentication", "2fa"], "type": "security", "importance": 0.95},
    "Remote Access": {"aliases": ["remote access", "remote login", "remote desktop", "rdp", "ssh access"], "type": "technology", "importance": 0.8},
    "Firewall": {"aliases": ["firewall", "packet filtering"], "type": "security", "importance": 0.85},
    "Password Policy": {"aliases": ["password policy", "password requirements", "password complexity"], "type": "policy", "importance": 0.8},
    "Software Installation": {"aliases": ["software installation", "software deployment", "app installation"], "type": "procedure", "importance": 0.7},
    "Security Incident": {"aliases": ["security incident", "security breach", "malware"], "type": "security", "importance": 0.9},
    "Data Backup": {"aliases": ["data backup", "backup", "data recovery"], "type": "procedure", "importance": 0.8},
    "Network Access": {"aliases": ["network access", "network connectivity", "lan access"], "type": "technology", "importance": 0.75},
}


class KGLiteService:
    def __init__(self):
        self.alias_to_concept: Dict[str, str] = {}
        for cname, meta in IT_CONCEPTS.items():
            for alias in meta["aliases"] + [cname]:
                self.alias_to_concept[alias.lower()] = cname

    def find_concepts_in_text(self, text_in: str) -> List[str]:
        text_lower = (text_in or "").lower()
        found: Set[str] = set()
        for alias, cname in self.alias_to_concept.items():
            if alias in text_lower:
                found.add(cname)
        return sorted(list(found))

    def rebuild_from_policies(self, db: Session) -> Dict[str, Any]:
        chunks: List[PolicyChunk] = db.query(PolicyChunk).all()
        if not chunks:
            return {"message": "No policy chunks found"}

        # Ensure concepts exist
        existing = {c.name: c for c in db.query(KnowledgeGraphConcept).all()}
        for cname, meta in IT_CONCEPTS.items():
            if cname not in existing:
                db.add(KnowledgeGraphConcept(
                    name=cname,
                    concept_type=meta.get("type", "technology"),
                    description=None,
                    aliases=meta.get("aliases", []),
                    importance_score=meta.get("importance", 0.7),
                    policy_chunks=[]
                ))
        db.commit()
        concepts_by_name = {c.name: c for c in db.query(KnowledgeGraphConcept).all()}

        # Scan chunks for concept mentions
        concept_mentions: Dict[int, Set[str]] = defaultdict(set)
        for ch in chunks:
            text_lower = (ch.content or "").lower()
            for alias, cname in self.alias_to_concept.items():
                if alias in text_lower:
                    concept_mentions[ch.id].add(cname)

        # Update policy_chunks fields
        for cname in concepts_by_name:
            concepts_by_name[cname].policy_chunks = []
        for chunk_id, cnames in concept_mentions.items():
            for cname in cnames:
                c = concepts_by_name.get(cname)
                if c:
                    lst = set(c.policy_chunks or [])
                    lst.add(chunk_id)
                    c.policy_chunks = sorted(list(lst))
        db.commit()

        # Co-occurrence relationships
        created = 0
        for chunk_id, cnames in concept_mentions.items():
            cnames = sorted(list(cnames))
            for i in range(len(cnames)):
                for j in range(i + 1, len(cnames)):
                    source = concepts_by_name[cnames[i]].id
                    target = concepts_by_name[cnames[j]].id
                    exists = db.execute(text(
                        "SELECT 1 FROM concept_relationships WHERE source_concept_id=:s AND target_concept_id=:t AND relationship_type='related_to'"
                    ), {"s": source, "t": target}).fetchone()
                    if not exists:
                        db.execute(text(
                            "INSERT INTO concept_relationships (source_concept_id, target_concept_id, relationship_type, weight, created_at) VALUES (:s, :t, 'related_to', :w, CURRENT_TIMESTAMP)"
                        ), {"s": source, "t": target, "w": 1.0})
                        created += 1
        db.commit()
        return {"concepts": len(concepts_by_name), "chunks_scanned": len(chunks), "relationships_created": created}

    def neighbors(self, db: Session, concept_id: int) -> List[int]:
        rows = db.execute(text(
            """
            SELECT target_concept_id FROM concept_relationships WHERE source_concept_id=:cid
            UNION
            SELECT source_concept_id FROM concept_relationships WHERE target_concept_id=:cid
            """
        ), {"cid": concept_id}).fetchall()
        return [r[0] for r in rows]

    def path_by_name(self, db: Session, source_name: str, target_name: str) -> List[str]:
        src = db.query(KnowledgeGraphConcept).filter(KnowledgeGraphConcept.name.ilike(f"%{source_name}%")).first()
        tgt = db.query(KnowledgeGraphConcept).filter(KnowledgeGraphConcept.name.ilike(f"%{target_name}%")).first()
        if not src or not tgt:
            return []
        if src.id == tgt.id:
            return [src.name]
        # BFS
        queue = deque([src.id])
        visited = {src.id}
        parent = {src.id: None}
        while queue:
            cur = queue.popleft()
            for nb in self.neighbors(db, cur):
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = cur
                    if nb == tgt.id:
                        path_ids = [nb]
                        while parent[path_ids[-1]] is not None:
                            path_ids.append(parent[path_ids[-1]])
                        path_ids.reverse()
                        # map ids to names
                        names = []
                        for cid in path_ids:
                            c = db.query(KnowledgeGraphConcept).get(cid)
                            names.append(c.name if c else str(cid))
                        return names
                    queue.append(nb)
        return []

kg_lite_service = KGLiteService()

