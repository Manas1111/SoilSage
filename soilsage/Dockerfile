FROM python:3.11-slim

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    SOILSAGE_LLM=groq

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Pre-download the embedding model at build time so the first visit isn't slow
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# Build the vector index at deploy time from data/processed/chunks.jsonl
# (keeps the 15 MB Chroma binary out of git)
RUN python -m src.build_index

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
