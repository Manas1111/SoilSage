"""RAG chain: retrieve -> guard -> prompt -> LLM -> answer + citations.

LLM backend is chosen by env var SOILSAGE_LLM:
  groq    (needs GROQ_API_KEY)      free, fast
  ollama  (local server running)    free, local
  none    (default)                 extractive fallback: returns best passages, no generation
"""
import json
import os
import re
import urllib.error
import urllib.request

from src.config import MIN_COVERAGE, MIN_BM25_SCORE, MIN_DENSE_SIM, TOP_K
from src.retriever import Retriever

SYSTEM = (
    "You are SoilSage, an agriculture advisor for Indian farmers and students. "
    "Answer ONLY from the numbered context passages; never use outside knowledge. "
    "Cite with plain square brackets like [1] or [2][3], right after the claim they support. "
    "Rules for accuracy: (1) Copy numbers and units exactly as written. "
    "(2) Keep every condition attached to its number (crop, variety, soil type, source of nutrient such as "
    "urea vs FYM). Never apply a condition or exception from one sentence to a different quantity. "
    "(3) If passages disagree or the exact answer is not stated, say so briefly instead of guessing. "
    "(4) If the context does not contain the answer, say you do not have enough information. "
    "Be short and practical. Passages from the US Midwest guide use lb/acre and may not apply to India; say so if used."
)
NO_INFO = ("I don't have enough information in my soil documents to answer this reliably. "
           "Try rephrasing, or consult your local Krishi Vigyan Kendra (KVK).")


def build_prompt(question: str, chunks: list[dict], soil_context: str = "") -> str:
    ctx = "\n\n".join(f"[{i}] ({c['source']}, p.{c['page']})\n{c['text']}"
                      for i, c in enumerate(chunks, 1))
    soil = f"Farmer's soil values: {soil_context}\n\n" if soil_context else ""
    return f"{soil}Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"


def _post(url, payload, headers):
    headers = {"User-Agent": "soilsage/0.1", "Accept": "application/json", **headers}
    req = urllib.request.Request(url, json.dumps(payload).encode(), headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:  # show the server's reason, not just the code
        raise RuntimeError(f"LLM API error {e.code}: {e.read().decode(errors='ignore')[:300]}") from None


def call_llm(prompt: str) -> str | None:
    backend = os.getenv("SOILSAGE_LLM", "none").lower()
    if backend == "groq":
        out = _post(
            "https://api.groq.com/openai/v1/chat/completions",
            {"model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), "temperature": 0.1,
             "messages": [{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}]},
            {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
             "Content-Type": "application/json"})
        return out["choices"][0]["message"]["content"]
    if backend == "ollama":
        out = _post(
            os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat",
            {"model": os.getenv("OLLAMA_MODEL", "llama3"), "stream": False,
             "options": {"temperature": 0.1},
             "messages": [{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}]},
            {"Content-Type": "application/json"})
        return out["message"]["content"]
    return None  # extractive fallback


class SoilSage:
    def __init__(self, use_dense: bool = True):
        self.retriever = Retriever(use_dense=use_dense)

    def _confident(self, query: str, hits: list[dict]) -> bool:
        if not hits:
            return False
        if self.retriever.coverage(query, hits) < MIN_COVERAGE:
            return False  # key terms of the question never appear in the documents
        if hits[0]["mode"] == "bm25":
            return hits[0]["score"] >= MIN_BM25_SCORE
        return hits[0]["norm"] >= MIN_DENSE_SIM

    def ask(self, question: str, soil_context: str = "", k: int = TOP_K) -> dict:
        query = f"{soil_context} {question}".strip()
        hits = self.retriever.search(query, k=k)
        sources = [{"n": i, "source": h["source"], "page": h["page"],
                    "score": round(h["score"], 3), "snippet": h["text"][:300]}
                   for i, h in enumerate(hits, 1)]
        if not self._confident(query, hits):
            return {"answer": NO_INFO, "sources": [], "grounded": False}

        answer = call_llm(build_prompt(question, hits, soil_context))
        if answer:  # normalize citation style (some models emit 【1】)
            answer = re.sub(r"[【\[]\s*(\d+)\s*[】\]]", r"[\1]", answer).strip()
        if answer is None:  # no LLM configured: show best passages with citations
            answer = "(No LLM configured; showing most relevant passages)\n\n" + "\n\n".join(
                f"[{s['n']}] {s['snippet']}..." for s in sources[:3])
        return {"answer": answer, "sources": sources, "grounded": True}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How to reclaim acidic soil with low pH?"
    res = SoilSage().ask(q)
    print(res["answer"], "\n")
    for s in res["sources"]:
        print(f"[{s['n']}] {s['source']} p.{s['page']}  score={s['score']}")