# app/prompt.py
from typing import List, Dict, Optional

def _format_retrieved_concise(retrieved: List[Dict[str, str]]) -> str:
    lines = []
    for d in retrieved:
        src = d.get("source", "unknown.txt")
        txt = d.get("text", "").strip()
        if not txt:
            txt = "(no summary available)"
        # single-line safe snippet
        single = " ".join(txt.splitlines())
        lines.append(f"--- Source: {src} ---\n{single}\n")
    return "\n".join(lines)

def build_prompt(question: str, retrieved_docs: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Strict prompt with explicit BEGIN/END markers. Model must only generate
    the ANSWER between BEGIN_ANSWER and END_ANSWER and then list SOURCES.
    """

    header = (
        "You are a helpful support assistant. Use ONLY the information in the CONTEXT below.\n"
        "DO NOT invent facts. If information is missing, reply exactly: \"I don't know based on the provided context.\"\n\n"
    )

    format_instructions = (
        "STRICT OUTPUT FORMAT (follow exactly):\n"
        "===BEGIN_ANSWER===\n"
        "<Short concise answer: 1-2 sentences>\n"
        "===END_ANSWER===\n"
        "SOURCES: (comma-separated filenames)\n\n"
        "Do NOT include any other text outside the markers.\n\n"
    )

    qblock = f"QUESTION: {question}\n\n"

    if retrieved_docs:
        context = "CONTEXT:\n" + _format_retrieved_concise(retrieved_docs) + "\n"
    else:
        context = "CONTEXT: (no documents available)\n\n"

    prompt = header + format_instructions + qblock + context + "Now provide the ANSWER following the STRICT OUTPUT FORMAT above.\n"
    return prompt
