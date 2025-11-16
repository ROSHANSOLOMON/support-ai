# scripts/debug_retriever.py
import os, sys, textwrap

# ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("Python:", sys.executable)
print("Project root:", PROJECT_ROOT)
print("sys.path[0]:", sys.path[0])

# import the Retriever from your app package
try:
    from app.retrieval import Retriever
except Exception as e:
    print("Failed to import Retriever from app.retrieval:", type(e).__name__, e)
    raise

def pretty_print_docs(docs):
    if not docs:
        print("Retriever returned 0 docs.")
        return
    for i, d in enumerate(docs):
        src = d.get("source", "unknown")
        txt = d.get("text", "") or ""
        snippet = " ".join(txt.strip().split())[:500]
        print(f"\n--- Doc #{i}  source: {src} ---")
        print(textwrap.fill(snippet, width=120))
        print("-" * 60)

def main():
    print("\\nInstantiating Retriever...")
    # try common constructor signatures
    try:
        r = Retriever()
    except TypeError:
        # try with explicit data_dir arg
        try:
            r = Retriever(docs_dir=os.path.join(PROJECT_ROOT, "data"))
        except Exception as e:
            print("Retriever instantiation failed:", type(e).__name__, e)
            raise
    except Exception as e:
        print("Retriever instantiation failed:", type(e).__name__, e)
        raise

    query = "How do I reset my password?"
    print("\\nQuery:", query)
    # attempt several common method names
    for method in ("retrieve", "query", "get_relevant_docs", "get_documents", "search"):
        if hasattr(r, method):
            try:
                docs = getattr(r, method)(query, top_k=6)
                print(f"Used retriever method: {method} -> returned {len(docs)} docs")
                pretty_print_docs(docs)
                return
            except TypeError:
                # try without top_k
                docs = getattr(r, method)(query)
                print(f"Used retriever method: {method} -> returned {len(docs)} docs")
                pretty_print_docs(docs)
                return
            except Exception as e:
                print(f"Method {method} raised {type(e).__name__}: {e}")
    # fallback: print loaded docs if available as attribute
    if hasattr(r, "_docs"):
        docs = getattr(r, "_docs")
        print(f"Fallback: r._docs length = {len(docs)}")
        pretty_print_docs(docs)
    else:
        print("No retrieval method matched and no _docs attribute found.")

if __name__ == "__main__":
    main()
