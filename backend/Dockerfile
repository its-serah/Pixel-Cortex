# Use slim Python image
FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install Ollama inside the container
RUN curl -fsSL https://ollama.com/install.sh | sh

# App setup
WORKDIR /app
COPY requirements.prod.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend code
COPY . /app

# Ensure start script is executable
RUN chmod +x /app/start.sh

# Env
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=127.0.0.1:11434

EXPOSE 8000

# Start Ollama and API
CMD ["bash", "/app/start.sh"]
