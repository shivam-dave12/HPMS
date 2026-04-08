# ─────────────────────────────────────────────────────────────────────────────
# Unified Quant Bot — Dockerfile
# Build:  docker build -t quant-bot .
# Run:    docker run --env-file .env --restart unless-stopped quant-bot
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install only what pip needs to compile (nothing native required for these deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Keeps Python from buffering stdout/stderr — log lines appear immediately
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Non-root user for security
RUN useradd --no-create-home --shell /bin/false botuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy the full project tree.
# .dockerignore should exclude: .env, __pycache__, *.pyc, *.log, .git
COPY . .

# Logs land in /app — make it writable by botuser
RUN chown -R botuser:botuser /app

USER botuser

# Entrypoint: start via the Telegram controller so /start launches the bot
# (switch to `python main.py` if you want the bot to start automatically)
CMD ["python", "telegram/controller.py"]
