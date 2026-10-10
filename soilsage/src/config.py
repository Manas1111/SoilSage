from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
CHUNKS_FILE = PROC_DIR / "chunks.jsonl"

# Chunking (characters; ~4 chars per token, so 2400 chars is about 600 tokens)
CHUNK_SIZE = 2400
CHUNK_OVERLAP = 300
MIN_CHUNK_CHARS = 200  # drop tiny/noisy chunks

# Retrieval / models
CHROMA_DIR = ROOT / "data" / "chroma"
EMBED_MODEL = "intfloat/multilingual-e5-small"   # swap to BAAI/bge-m3 later
TOP_K = 8
# Guard thresholds: below these, answer "not enough information"
MIN_BM25_SCORE = 4.0      # raw BM25 score of best chunk
MIN_DENSE_SIM = 0.80      # cosine sim of best chunk (e5 sims run high)
MIN_COVERAGE = 0.4    # share of question terms (IDF-weighted) that must co-occur in one top chunk
