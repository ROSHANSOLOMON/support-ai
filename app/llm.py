# app/llm.py
"""
Lightweight wrapper around llama-cpp-python's Llama.

Features:
- Loads a local GGUF model from models/
- Thread-safe calls via a lock to avoid race conditions
- Accepts both positional and keyword args in __call__ so older code like llm(q, enriched) works
- answer(...) accepts stop tokens and forwards them to the underlying Llama call
- Safe close(), context-manager and __del__ handling to release native resources
"""

import os
import threading
from typing import Any, Optional, List

from llama_cpp import Llama


class LLM:
    def __init__(
        self,
        model_filename: str = "phi-2.Q4_K_M.gguf",
        n_threads: int = 6,
        n_ctx: int = 2048,
        n_batch: int = 256,
    ):
        model_path = os.path.join("models", model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Please download the model and place it in the models/ folder."
            )

        print(f"Loading local model from: {model_path}")
        # adjust n_threads to match your CPU cores. Reduce if CPU is overloaded.
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            n_gpu_layers=0,  # force CPU-only inference
            verbose=False,
        )

        # lock to avoid concurrent calls causing internal race conditions
        self._call_lock = threading.Lock()
        print("Model loaded successfully!")

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """
        Accept any positional/keyword args so calls like llm(q, enriched) work.

        Behavior:
          - If positional args: treat args[0] as prompt, args[1] as enriched (if present)
          - Keyword args 'prompt' and 'enriched' also supported
          - All remaining kwargs forwarded to answer()
        """
        prompt: Optional[str] = None
        enriched = None

        if len(args) >= 1:
            prompt = args[0]
        if len(args) >= 2:
            enriched = args[1]

        # allow prompt/enriched via kwargs as well (kwargs takes precedence for enriched)
        if prompt is None:
            prompt = kwargs.pop("prompt", None)
        enriched = kwargs.pop("enriched", enriched)

        return self.answer(prompt=prompt, enriched=enriched, **kwargs)

    def answer(
        self,
        prompt: str,
        enriched: Optional[Any] = None,
        max_tokens: int = 200,
        temperature: float = 0.0,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate and return text for the provided prompt.

        Arguments:
          prompt: main prompt string (required)
          enriched: optional string or object to augment prompt (if string, prepended)
          max_tokens, temperature, top_p: generation parameters
          stop: optional list of stop-token strings (e.g. ["===END_ANSWER==="]) to halt generation

        Returns:
          Generated text (str) — robustly extracted from llama-cpp-python response.
        """
        if prompt is None:
            raise ValueError("No prompt provided to LLM.answer()")

        # incorporate enriched data (conservative default: prepend if string)
        if enriched:
            if isinstance(enriched, str):
                full_prompt = enriched + "\n" + prompt
            else:
                full_prompt = f"{prompt}\n\n[ENRICHED]\n{str(enriched)}"
        else:
            full_prompt = prompt

        with self._call_lock:
            # llama-cpp-python accepts stop as a list of strings in many versions.
            # If your version expects a different param type, adapt accordingly.
            resp = self.llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )

        # robustly extract text from response (handle different wrapper return shapes)
        text = ""
        try:
            if isinstance(resp, dict):
                # common shape: {'choices': [{'text': '...'}], ...}
                choices = resp.get("choices")
                if choices and isinstance(choices, list):
                    first = choices[0]
                    # some versions use 'text', others might use 'message' etc.
                    if isinstance(first, dict):
                        text = first.get("text") or first.get("message") or ""
                    else:
                        text = str(first)
                else:
                    # maybe a top-level 'text'
                    text = resp.get("text") or ""
            else:
                # fallback: convert to string
                text = str(resp)
        except Exception:
            # last-resort fallback
            try:
                text = str(resp)
            except Exception:
                text = ""

        return (text or "").strip()

    def close(self) -> None:
        """Attempt to release native resources cleanly."""
        try:
            close_fn = getattr(self.llm, "close", None)
            if callable(close_fn):
                close_fn()
            # remove reference to allow GC and finalizers
            del self.llm
        except Exception:
            # swallow errors to avoid noisy tracebacks on interpreter shutdown
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
