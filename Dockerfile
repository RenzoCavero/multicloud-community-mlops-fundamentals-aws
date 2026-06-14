# Imagen de la API de inferencia (FastAPI). Se construye en GitHub Actions,
# se sube a ECR y se ejecuta en EC2 con Docker.
FROM python:3.11-slim

WORKDIR /app

# Dependencias primero (mejor cache de capas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo de la app
COPY src/ ./src/

# src es la raiz de fuentes (imports planos: import config, import inference)
ENV PYTHONPATH=/app/src
# Region por defecto; el modelo se descarga del registry en runtime
ENV AWS_REGION=us-east-1

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
