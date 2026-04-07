FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better caching
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY models.py .
COPY client.py .
COPY inference.py .
COPY openenv.yaml .
COPY server/ ./server/

# Expose the default port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
