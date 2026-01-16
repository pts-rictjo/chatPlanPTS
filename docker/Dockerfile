FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV OLLAMA_HOST=http://127.0.0.1:11434

RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    ca-certificates \
    pciutils \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Installera Ollama (auto GPU/CPU)
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY app/query_llm.py .
COPY requirements.txt .
COPY docker/entrypoint.sh .

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]
