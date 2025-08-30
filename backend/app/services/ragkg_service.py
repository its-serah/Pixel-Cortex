import os
import re
import json
import glob
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# Fast, lightweight RAG using TF-IDF (no heavy embeddings)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib

# Lightweight KG using NetworkX
import networkx as nx

# ---------- Paths & Helpers ----------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = ensure_dir(os.path.join(BASE_DIR, "data"))
RAG_DIR = ensure_dir(os.path.join(DATA_DIR, "rag"))
KG_DIR = ensure_dir(os.path.join(DATA_DIR, "kg"))

TFIDF_PATH = os.path.join(RAG_DIR, "tfidf_vectorizer.joblib")
MATRIX_PATH = os.path.join(RAG_DIR, "tfidf_matrix.joblib")
CHUNKS_PATH = os.path.join(RAG_DIR, "chunks.jsonl")

CONCEPTS_PATH = os.path.join(KG_DIR, "concepts.jsonl")
EDGES_PATH = os.path.join(KG_DIR, "edges.jsonl")

# ---------- Chunking ----------

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+max_chars]
        chunks.append(chunk)
        i += max(1, max_chars - overlap)
    return chunks

# ---------- File Loading ----------

def load_text_from_file(path: str) -> str:
    try:
        if path.lower().endswith(".pdf"):
            try:
                import PyPDF2  # optional
                text = []
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text.append(page.extract_text() or "")
                return "\n".join(text)
            except Exception:
                return ""  # skip pdf if lib not available
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        return ""

# ---------- Concepts & Relationships ----------

IT_CONCEPTS: Dict[str, Dict[str, Any]] = {
    "VPN": {
        "aliases": ["vpn", "virtual private network", "remote access vpn", "site-to-site vpn"],
        "type": "technology",
    },
    "MFA": {
        "aliases": ["mfa", "multi-factor authentication", "two-factor authentication", "2fa", "multifactor"],
        "type": "security",
    },
    "Remote Access": {
        "aliases": ["remote access", "remote login", "remote desktop", "rdp", "ssh access"],
        "type": "technology",
    },
    "Firewall": {
        "aliases": ["firewall", "network firewall", "packet filtering"],
        "type": "security",
    },
    "Password Policy": {
        "aliases": ["password policy", "password requirements", "password complexity", "password rules"],
        "type": "policy",
    },
    "Security Incident": {
        "aliases": ["security incident", "security breach", "malware", "attack", "phishing"],
        "type": "security",
    },
    "Network Access": {
        "aliases": ["network access", "network connectivity", "lan access"],
        "type": "technology",
    },
}

RELATION_PATTERNS = {
    "requires": [r"(\b\w+(?:\s+\w+)*)\s+requires?\s+(\b\w+(?:\s+\w+)*)"],
    "depends_on": [r"(\b\w+(?:\s+\w+)*)\s+depends?\s+on\s+(\b\w+(?:\s+\w+)*)"],
    "enables": [r"(\b\w+(?:\s+\w+)*)\s+enables?\s+(\b\w+(?:\s+\w+)*)"],
    "related_to": [r"(\b\w+(?:\s+\w+)*)\s+(?:and|with)\s+(\b\w+(?:\s+\w+)*)"],
}

# ---------- Data Models ----------

@dataclass
class Chunk:
    id: str
    document_path: str
    document_title: str
    content: str
    order: int

# ---------- RAG Index ----------

class RAGIndex:
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.chunks: List[Chunk] = []
        self._loaded = False

    def load(self):
        if os.path.exists(TFIDF_PATH) and os.path.exists(MATRIX_PATH) and os.path.exists(CHUNKS_PATH):
            self.vectorizer = joblib.load(TFIDF_PATH)
            self.matrix = joblib.load(MATRIX_PATH)
            self.chunks = []
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    self.chunks.append(Chunk(**obj))
            self._loaded = True
        return self._loaded

    def save(self):
        if self.vectorizer is not None and self.matrix is not None and self.chunks:
            joblib.dump(self.vectorizer, TFIDF_PATH)
            joblib.dump(self.matrix, MATRIX_PATH)
            with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
                for c in self.chunks:
                    f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")

    def index_dir(self, policies_dir: str) -> Dict[str, Any]:
        paths = []
        for ext in ("*.md", "*.txt", "*.pdf"):
            paths.extend(glob.glob(os.path.join(policies_dir, "**", ext), recursive=True))
        docs: List[Chunk] = []
        for p in paths:
            text = load_text_from_file(p)
            if not text:
                continue
            title = os.path.basename(p)
            pieces = chunk_text(text)
            for i, piece in enumerate(pieces):
                cid = hashlib.md5(f"{p}:{i}".encode()).hexdigest()
                docs.append(Chunk(id=cid, document_path=p, document_title=title, content=piece, order=i))
        if not docs:
            # clear previous index if any
            if os.path.exists(CHUNKS_PATH):
                os.remove(CHUNKS_PATH)
            if os.path.exists(TFIDF_PATH):
                os.remove(TFIDF_PATH)
            if os.path.exists(MATRIX_PATH):
                os.remove(MATRIX_PATH)
            self.vectorizer = None
            self.matrix = None
            self.chunks = []
            self._loaded = False
            return {"indexed_chunks": 0, "documents": 0}
        corpus = [d.content for d in docs]
        vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
        matrix = vectorizer.fit_transform(corpus)
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunks = docs
        self.save()
        self._loaded = True
        return {"indexed_chunks": len(docs), "documents": len(set(d.document_path for d in docs))}

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self._loaded:
            self.load()
        if not self.vectorizer or self.matrix is None or not self.chunks:
            return []
        qv = self.vectorizer.transform([query])
        scores = linear_kernel(qv, self.matrix).flatten()
        top_idx = scores.argsort()[::-1][:k]
        results = []
        for idx in top_idx:
            c = self.chunks[idx]
            results.append({
                "chunk_id": c.id,
                "document_title": c.document_title,
                "document_path": c.document_path,
                "content": c.content,
                "score": float(scores[idx])
            })
        return results

# ---------- KG Builder ----------

class KGBuilder:
    def __init__(self):
        self.concepts: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Tuple[str, str, str]] = []  # (src, dst, rel)
        self.graph: Optional[nx.DiGraph] = None

    def _map_to_concept(self, text: str) -> Optional[str]:
        tl = text.lower().strip()
        for name, data in IT_CONCEPTS.items():
            for alias in data["aliases"]:
                if alias in tl or tl in alias:
                    return name
        return None

    def extract_concepts(self, text: str) -> List[str]:
        found = set()
        tl = text.lower()
        for name, data in IT_CONCEPTS.items():
            for alias in data["aliases"]:
                if alias in tl:
                    found.add(name)
                    break
        return list(found)

    def extract_relationships(self, text: str) -> List[Tuple[str, str, str]]:
        rels = []
        for rel, patterns in RELATION_PATTERNS.items():
            for pat in patterns:
                for m in re.finditer(pat, text, flags=re.IGNORECASE):
                    src = self._map_to_concept(m.group(1))
                    dst = self._map_to_concept(m.group(2))
                    if src and dst and src != dst:
                        rels.append((src, dst, rel))
        return rels

    def build_from_chunks(self, chunks: List[Chunk]) -> Dict[str, Any]:
        self.concepts = {}
        self.edges = []
        for ch in chunks:
            concepts = self.extract_concepts(ch.content)
            for c in concepts:
                self.concepts.setdefault(c, {"type": IT_CONCEPTS[c]["type"], "mentions": 0})
                self.concepts[c]["mentions"] += 1
            # intra-chunk relationships
            self.edges.extend(self.extract_relationships(ch.content))
        # persist
        with open(CONCEPTS_PATH, "w", encoding="utf-8") as f:
            for name, data in self.concepts.items():
                f.write(json.dumps({"name": name, **data}) + "\n")
        with open(EDGES_PATH, "w", encoding="utf-8") as f:
            for src, dst, rel in self.edges:
                f.write(json.dumps({"source": src, "target": dst, "relationship": rel}) + "\n")
        return {"concepts": len(self.concepts), "relationships": len(self.edges)}

    def load_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        if os.path.exists(CONCEPTS_PATH):
            with open(CONCEPTS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    o = json.loads(line)
                    G.add_node(o["name"], **{k: v for k, v in o.items() if k != "name"})
        if os.path.exists(EDGES_PATH):
            with open(EDGES_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    o = json.loads(line)
                    G.add_edge(o["source"], o["target"], relationship=o["relationship"]) 
        self.graph = G
        return G

    def query_related(self, concepts: List[str], max_hops: int = 2) -> Dict[str, Any]:
        if not self.graph:
            self.load_graph()
        if not self.graph or len(self.graph) == 0:
            return {"related": [], "hops": []}
        related = set()
        hops: List[Dict[str, Any]] = []
        for c in concepts:
            if c not in self.graph:
                continue
            # BFS up to max_hops
            visited = {c}
            frontier = [(c, 0)]
            while frontier:
                node, depth = frontier.pop(0)
                if depth >= max_hops:
                    continue
                for nbr in self.graph.successors(node):
                    if nbr not in visited:
                        visited.add(nbr)
                        related.add(nbr)
                        hops.append({"from": node, "to": nbr, "depth": depth+1, "relationship": self.graph.edges[node, nbr].get("relationship", "related_to")})
                        frontier.append((nbr, depth+1))
        return {"related": sorted(list(related)), "hops": hops}

# ---------- Orchestrator ----------

class RAGKG:
    def __init__(self):
        self.index = RAGIndex()
        self.kg = KGBuilder()

    def index_policies(self, policies_dir: str) -> Dict[str, Any]:
        res = self.index.index_dir(policies_dir)
        # build KG immediately for speed of use
        kg_res = self.kg.build_from_chunks(self.index.chunks)
        return {"rag": res, "kg": kg_res}

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        return self.index.search(query, k=k)

    def build_kg(self) -> Dict[str, Any]:
        # Rebuild from current chunks
        if not self.index._loaded:
            self.index.load()
        return self.kg.build_from_chunks(self.index.chunks)

    def query_graph(self, concepts: List[str], max_hops: int = 2) -> Dict[str, Any]:
        return self.kg.query_related(concepts, max_hops=max_hops)

