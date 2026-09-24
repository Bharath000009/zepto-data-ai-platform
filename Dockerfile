FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by some wheels (torch, chromadb)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY support_assistant/ ./support_assistant/

# Set working directory to the module
WORKDIR /app/support_assistant

# Expose the FastAPI port
EXPOSE 7860

# Serve the FastAPI app (default MOCK_LLM=1 — fully offline)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]