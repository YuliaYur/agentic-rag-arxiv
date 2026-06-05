# One image serving both the FastAPI API and the Streamlit UI (the compose `ui`
# service just overrides the command). Kept CPU-only and lean.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf-cache \
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only PyTorch first — otherwise sentence-transformers pulls the multi-GB CUDA wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the package + serve/ui extras from the source tree.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve,ui]"

# Runtime files that aren't part of the importable package: the index-loader script,
# the committed index fixture, and the Streamlit app.
COPY scripts ./scripts
COPY ui ./ui
COPY eval/fixtures ./eval/fixtures

EXPOSE 8000 8501
# Default = the API; the `ui` service overrides this with the streamlit command.
CMD ["uvicorn", "agentic_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
