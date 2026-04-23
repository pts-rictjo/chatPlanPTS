FROM python:3.11-slim

# --- miljö ---
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- systemberoenden ---
RUN apt-get update && apt-get install -y \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# --- skapa user ---
RUN useradd -m -u 10001 appuser

WORKDIR /app

# --- installera dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- kopiera app ---
COPY app/ ./app/

# --- permissions ---
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# --- viktigt: streamlit korrekt ---
CMD ["streamlit", "run", "app/query_llm.py", "--server.address=0.0.0.0", "--server.port=8501"]
