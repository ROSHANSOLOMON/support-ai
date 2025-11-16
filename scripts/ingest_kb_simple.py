# scripts/ingest_kb_simple.py
import os, json, math
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EMB_PATH = DATA_DIR / "embeddings.npy"
META_PATH = DATA_DIR / "metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast sentence-transformer

def load_text_files(data_dir: Path):
    files = sorted([p for p in data_dir.glob("*.txt")])
    docs = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        # one-line summary (first non-empty line)
        summary = ""
        for line in text.splitlines():
            if line.strip():
                summary = line.strip()
                break
        docs.append({"source": p.name, "text": summary, "raw": text})
    return docs

def chunk_iter(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]

def main():
    docs = load_text_files(DATA_DIR)
    print(f"Found {len(docs)} docs.")
    if len(docs) == 0:
        print("No .txt files found in data/. Please put your KB .txt files there.")
        return

    print("Loading sentence-transformers model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    # compute embeddings in batches
    texts = [d["raw"] or d["text"] for d in docs]
    batch_size = 32
    embs = []
    for batch in chunk_iter(texts, batch_size):
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embs.append(emb)
    embs = np.vstack(embs)
    print("Embeddings shape:", embs.shape)

    # save embeddings and metadata
    np.save(EMB_PATH, embs)
    print("Saved embeddings to", EMB_PATH)

    # store metadata as simple list of dicts
    meta = [{"source": d["source"], "text": d["text"]} for d in docs]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Saved metadata to", META_PATH)
    print("Done.")
    
if __name__ == '__main__':
    main()
