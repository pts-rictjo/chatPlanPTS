FROM python:3.12-slim

WORKDIR /app

# Systempaket som ofta behövs av ChromaDB,
# numpy, sentence-transformers etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installera Pythonberoenden
COPY ../requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Kopiera applikationen
COPY ./app ./app/
COPY ./helper ./helper/
COPY ./static ./static/
COPY ./public ./public/

# Mountpunkter för Kubernetes
RUN mkdir -p /app/data
RUN mkdir -p /app/chroma_db

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh    
ENTRYPOINT ["./entrypoint.sh"]

# Miljö
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
