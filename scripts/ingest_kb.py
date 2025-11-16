# scripts/ingest_kb.py
import glob
import os
from app.retrieval import Retriever

def load_kb(kb_dir="data/kb"):
    files = sorted(glob.glob(os.path.join(kb_dir, "*.txt")))
    docs = []
    for i, f in enumerate(files):
        with open(f, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
        docs.append({"id": i, "text": text, "source": os.path.basename(f)})
    return docs

if __name__ == "__main__":
    r = Retriever()
    docs = load_kb()
    print(f"Found {len(docs)} docs. Adding to index...")
    r.add(docs)
    print("Index built. Files written: data/embeddings.npy and data/metadata.npy")
