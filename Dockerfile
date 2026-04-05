FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal runtime tooling and TLS roots
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Install uv and project dependencies (from lockfile when available)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

# Runtime directories (persistent data + optional inbox)
RUN mkdir -p /data /inbox

# Daemon mode
CMD ["uv", "run", "python", "run.py", "--daemon", "--interval", "0.5"]
