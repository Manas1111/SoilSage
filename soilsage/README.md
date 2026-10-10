# 🌱 SoilSage: RAG-based Soil Health & Crop Advisory

Ask a soil or crop question, optionally enter soil-test values (pH, EC, organic carbon, N, P, K), and get a short answer **with source and page citations** drawn from Indian agricultural documents. If the documents don't contain the answer, SoilSage says so instead of guessing.

![SoilSage demo](docs/screenshot.png)

## Features
- **Hybrid retrieval:** BM25 keyword search + dense vector search (`multilingual-e5-small`, ChromaDB), merged with Reciprocal Rank Fusion.
- **Cited answers:** every claim points to `[n]` = file name + page; the UI lists the source passages.
- **Hallucination guard:** refuses off-topic questions when the question's key terms don't co-occur in any retrieved passage.
- **Soil-test interpretation:** rule-based ratings (pH class, salinity, low/medium/high N-P-K and organic carbon) that are passed to the model as context.
- **Pluggable LLM:** Groq API or local Ollama; works without an LLM by showing the retrieved passages.
- **Streamlit UI.**

## Architecture
```
PDFs -> clean -> chunk (~600 tokens, overlap) -> chunks.jsonl
                                    |                |
                              embeddings          BM25 index
                              (ChromaDB)               |
 question + soil values --> hybrid retrieval (RRF) <---+
                                    |
                 coverage guard --> "not enough information"
                                    |
              prompt with numbered passages --> LLM --> answer + [n] citations
```

## Tech stack
Python, ChromaDB, sentence-transformers (`intfloat/multilingual-e5-small`), rank-bm25, Streamlit, Groq / Ollama.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Put text-based PDFs in data/raw/ (see data/SOURCES.md)
python -m src.ingest          # PDFs -> data/processed/chunks.jsonl
python -m src.build_index     # embeddings -> data/chroma/
python -m src.rag "How to reclaim saline and alkali soils?"

# optional: LLM-written answers
export SOILSAGE_LLM=groq GROQ_API_KEY=your_key GROQ_MODEL=<a Groq chat model>
python -m streamlit run app/streamlit_app.py
```

## Knowledge base used in development
Six public documents: PAU Punjab *Package of Practices* (Kharif and Rabi), a Rajasthan university *Soil Fertility and Nutrient Management* manual, an ICAR commercial-crops package, and a Midwest US vegetable fertility guide (US units, used as a secondary source). Extracted text chunks and the vector index are included so the demo runs; the original PDFs are not. All content belongs to the respective publishers.

## Known limitations
- **Ranking:** a question like "nitrogen dose for wheat" can rank passages that merely mention wheat above the page with the fertilizer table. A cross-encoder reranker is the planned fix.
- **LLM errors:** in testing, the model once attached a "half dose" rule that belongs to manures to the nitrogen dose. A stricter prompt fixed that case, but generated answers should always be checked against the cited page.
- **Small corpus:** gaps exist (for example micronutrients for vegetables).
- **Not evaluated yet:** no quantitative retrieval/answer metrics so far.

## Roadmap
- [ ] Evaluation set with retrieval hit rate and guard accuracy
- [ ] Cross-encoder reranker
- [ ] Crop recommendation model exposed as an LLM tool
- [ ] Hindi questions
- [ ] Embedding fine-tuning on soil Q&A pairs

*Advice is generated from retrieved documents and is not a substitute for local agronomist or Soil Health Card recommendations.*
