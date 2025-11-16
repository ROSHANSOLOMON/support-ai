# scripts/run_query_llm.py  (one-file replacement)
import time
from app.llm import LLM
from app.retrieval import Retriever
from app.prompt import build_prompt

def main():
    # tuned for responsiveness
    llm = LLM(model_filename="phi-2.Q4_K_M.gguf", n_threads=2, n_ctx=1024, n_batch=128)
    retr = Retriever()

    try:
        while True:
            q = input("Question> ").strip()
            if not q:
                continue

            docs = retr.retrieve(q, top_k=4)
            prompt = build_prompt(q, retrieved_docs=docs)

            print(f"\nSelected sources: {', '.join(d.get('source') for d in docs)}\n")

            t0 = time.time()
            ans = llm.answer(
                prompt=prompt,
                max_tokens=80,
                temperature=0.0,
                top_p=0.5,
                stop=["SOURCES:", "===END_ANSWER==="],
            )
            elapsed = time.time() - t0

            # Robust parsing
            raw = ans.strip()
            if "SOURCES:" in raw:
                extracted = raw.split("SOURCES:")[0].strip()
            else:
                start = raw.find("===BEGIN_ANSWER===")
                end = raw.find("===END_ANSWER===")
                if start != -1 and end != -1 and end > start:
                    extracted = raw[start + len("===BEGIN_ANSWER==="):end].strip()
                else:
                    for marker in ("\n===BEGIN_ANSWER===", "\n<Short concise answer:", "\n===END_ANSWER==="):
                        if marker in raw:
                            extracted = raw.split(marker)[0].strip()
                            break
                    else:
                        extracted = raw
            # clean template noise
            lines = []
            for line in extracted.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("<Short concise answer:") or line.startswith("===BEGIN_ANSWER===") or line.startswith("===END_ANSWER==="):
                    continue
                lines.append(line)
            extracted = "\n".join(lines).strip()

            print("\n==== ANSWER (synthesized) ====\n")
            print((extracted or "I don't know based on the provided context.") + "\n")
            print("==== SOURCES (retrieved) ====\n")
            for i, d in enumerate(docs):
                print(f"[{i}] {d.get('source')}   (snippet: {d.get('text')})")
            print(f"\n(Generation time: {elapsed:.1f}s)\n")

    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
    finally:
        llm.close()

if __name__ == "__main__":
    main()
