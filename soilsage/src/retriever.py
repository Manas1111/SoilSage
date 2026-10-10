"""Retrieval: BM25 (works anywhere) + optional dense (Chroma) + Reciprocal Rank Fusion.

Dense search needs the embedding model from Hugging Face, so build the index
with `python -m src.build_index` on Colab/local, then copy data/chroma/ back.
"""
import json
import re

from rank_bm25 import BM25Okapi

from src.config import CHROMA_DIR, CHUNKS_FILE, EMBED_MODEL

_TOKEN = re.compile(r"\w+", re.UNICODE)


STOP = set("""a an the is are was were be been of to in on for and or with by at from as it this that
these those how what when where which who whom why do does did can could should would will i my me we our you
your about into than then so if not no yes any some more most much many very also per use used using get give
tell please""".split())


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# Domain synonyms / spelling variants -> one canonical form
SYN = {"reclaim": "reclamation", "reclaiming": "reclamation", "acidic": "acid", "acidity": "acid",
       "alkaline": "alkali", "alkalinity": "alkali", "alkaline-sodic": "sodic", "salinity": "saline",
       "salt": "saline", "salty": "saline", "fertiliser": "fertilizer", "fertilisers": "fertilizer",
       "manure": "fyc", "compost": "fyc", "lime": "liming", "gypsum": "gypsum", "nitrogenous": "nitrogen",
       "phosphatic": "phosphorus", "potassic": "potassium", "organic": "organic"}
_SUFFIX = ("ations", "ation", "ings", "ing", "ies", "es", "s")


def _stem(t: str) -> str:
    t = SYN.get(t, t)
    for suf in _SUFFIX:
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            t = t[: -len(suf)]
            break
    return t


def content_terms(text: str) -> list[str]:
    return [_stem(t) for t in tokenize(text) if t not in STOP and len(t) > 1]


def load_chunks() -> list[dict]:
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


class Retriever:
    def __init__(self, use_dense: bool = True):
        self.chunks = load_chunks()
        self.by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([content_terms(c["text"]) for c in self.chunks])
        self.tokens = [set(content_terms(c["text"])) for c in self.chunks]
        import math
        self._unseen_idf = math.log(len(self.chunks) + 1)
        self.dense = None
        if use_dense and CHROMA_DIR.exists():
            try:
                import chromadb
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(EMBED_MODEL)
                client = chromadb.PersistentClient(path=str(CHROMA_DIR))
                self.dense = client.get_collection("soil")
            except Exception as e:  # missing model / index -> BM25 only
                print(f"[retriever] dense disabled: {e}")

    def _bm25(self, query: str, k: int):
        scores = self.bm25.get_scores(content_terms(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i]["id"], float(scores[i])) for i in order]

    def _dense(self, query: str, k: int):
        emb = self.model.encode([f"query: {query}"], normalize_embeddings=True).tolist()
        res = self.dense.query(query_embeddings=emb, n_results=k)
        # Chroma returns cosine distance; convert to similarity
        return [(i, 1 - d) for i, d in zip(res["ids"][0], res["distances"][0])]

    def search(self, query: str, k: int = 5, pool: int = 20) -> list[dict]:
        """Returns top-k chunks, each with 'score' (higher=better) and 'mode'."""
        bm = self._bm25(query, pool)
        if self.dense is None:
            top = bm[:k]
            max_s = max((s for _, s in top), default=0) or 1.0
            return [{**self.by_id[i], "score": s, "norm": s / max_s, "mode": "bm25"}
                    for i, s in top]

        dn = self._dense(query, pool)
        fused: dict[str, float] = {}
        for ranked in (bm, dn):  # Reciprocal Rank Fusion
            for rank, (i, _) in enumerate(ranked):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
        dense_sim = dict(dn)
        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{**self.by_id[i], "score": s, "norm": dense_sim.get(i, 0.0), "mode": "hybrid"}
                for i, s in top]

    def coverage(self, query: str, hits: list[dict]) -> float:
        """Best single-chunk IDF-weighted coverage of the query's content terms (0..1).
        Terms must co-occur in ONE chunk, so scattered common words from an
        off-topic question can't add up to a pass."""
        terms = set(content_terms(query))
        if not terms or not hits:
            return 0.0
        idx = {c["id"]: i for i, c in enumerate(self.chunks)}
        w = lambda t: max(self.bm25.idf.get(t, self._unseen_idf), 0.5)
        total = sum(w(t) for t in terms)
        return max(sum(w(t) for t in terms if t in self.tokens[idx[h["id"]]]) / total
                   for h in hits)
