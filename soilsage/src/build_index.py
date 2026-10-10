"""Embed all chunks into ChromaDB. Run on Colab/local (needs Hugging Face access).

Run: python -m src.build_index
"""
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import CHROMA_DIR, EMBED_MODEL
from src.retriever import load_chunks


def main(batch: int = 64):
    chunks = load_chunks()
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection("soil")
    except Exception:
        pass
    col = client.create_collection("soil", metadata={"hnsw:space": "cosine"})

    for i in tqdm(range(0, len(chunks), batch), desc="Embedding"):
        part = chunks[i:i + batch]
        emb = model.encode([f"passage: {c['text']}" for c in part],
                           normalize_embeddings=True).tolist()
        col.add(ids=[c["id"] for c in part], embeddings=emb,
                documents=[c["text"] for c in part],
                metadatas=[{"source": c["source"], "page": c["page"]} for c in part])
    print(f"Indexed {col.count()} chunks -> {CHROMA_DIR}")


if __name__ == "__main__":
    main()
