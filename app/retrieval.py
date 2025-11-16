# app/retrieval.py
import os, json, math, re
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

DATA_DIR = Path("data")
EMB_FILE = DATA_DIR / "embeddings.npy"
META_FILE_JSON = DATA_DIR / "metadata.json"

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # assume a and b are 1D numpy arrays
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

class Retriever:
    """
    Retriever that uses precomputed embeddings (if available) and falls back to a
    keyword-based retriever otherwise.

    - If embeddings.npy + metadata.json exist, uses them for semantic search.
    - metadata.json: list of {"source": filename, "text": summary}
    """

    def __init__(self, docs_dir: Optional[str] = None):
        self.docs_dir = Path(docs_dir) if docs_dir else DATA_DIR
        self._docs = self._load_docs_list()
        # load embeddings if present
        self.embeddings = None
        self.metadata = None
        if EMB_FILE.exists() and META_FILE_JSON.exists():
            try:
                self.embeddings = np.load(EMB_FILE)
                with open(META_FILE_JSON, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                # ensure lengths match
                if len(self.metadata) != self.embeddings.shape[0]:
                    print("Warning: metadata length != embeddings count. Ignoring embeddings.")
                    self.embeddings = None
                    self.metadata = None
            except Exception as e:
                print("Failed to load embeddings/metadata, falling back to keyword retrieval:", e)
                self.embeddings = None
                self.metadata = None

    def _load_docs_list(self) -> List[Dict[str, Any]]:
        docs = []
        if not self.docs_dir.exists():
            return docs
        for p in sorted(self.docs_dir.glob("*.txt")):
            try:
                raw = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                raw = ""
            summary = ""
            for line in raw.splitlines():
                if line.strip():
                    summary = line.strip()
                    break
            docs.append({"source": p.name, "text": summary, "raw": raw, "norm": _normalize(raw)})
        return docs

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # If we have embeddings + metadata, perform semantic search
        if self.embeddings is not None and self.metadata is not None:
            try:
                # lazy import to avoid heavy dependency when unused
                from sentence_transformers import SentenceTransformer
            except Exception:
                # cannot compute query embedding, fallback to keyword
                return self._keyword_retrieve(query, top_k)

            model = SentenceTransformer("all-MiniLM-L6-v2")
            q_emb = model.encode(query, convert_to_numpy=True)
            # compute cosine similarities
            sims = np.dot(self.embeddings, q_emb) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-12)
            # get top_k indices
            topk_idx = list(reversed(np.argsort(sims)))[0:top_k]
            results = []
            for idx in topk_idx:
                meta = self.metadata[idx]
                results.append({"source": meta.get("source"), "text": meta.get("text")})
            return results

        # otherwise fallback
        return self._keyword_retrieve(query, top_k)

    def _keyword_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = _normalize(query)
        q_tokens = [tok for tok in re.findall(r"\w+", q) if len(tok) > 1]
        if not q_tokens:
            return self._docs[:top_k]
        scored = []
        for d in self._docs:
            text = d.get("norm", "")
            score = sum(text.count(tok) for tok in q_tokens)
            fname = d.get("source", "").lower()
            score += sum(1 for tok in q_tokens if tok in fname)
            if score > 0:
                scored.append((score, d))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return [d for _, d in scored[:top_k]]
        return self._docs[:top_k]

    # compatibility names
    def query(self, q: str, k: int = 5):
        return self.retrieve(q, top_k=k)

    def refresh(self):
        self._docs = self._load_docs_list()
