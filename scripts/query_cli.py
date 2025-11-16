# scripts/query_cli.py
from app.retrieval import Retriever

if __name__ == "__main__":
    r = Retriever()
    print("Type a question (or 'exit')")
    
    while True:
        q = input("Query> ").strip()
        if q.lower() in ("exit", "quit"):
            break

        results = r.query(q, k=4)
        print("\nTop results:\n")
        for i, doc in enumerate(results):
            print(f"[{i}]  source={doc['source']}  score={doc['score']}\n")
