FROM python:3.12-slim

# System dependencies (ffmpeg, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source and project files
COPY pyproject.toml README.md ./
COPY src/ src/
COPY .env.example .env.example

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Create artifacts directory
RUN mkdir -p /app/artifacts

EXPOSE 8000

CMD ["uvicorn", "video_agent.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
