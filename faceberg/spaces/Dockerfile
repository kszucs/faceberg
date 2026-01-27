FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install faceberg from GitHub (using archive URL to avoid needing git)
RUN uv pip install --system --no-cache https://github.com/kszucs/faceberg/archive/refs/heads/main.tar.gz

# Expose HuggingFace Spaces default port
EXPOSE 7860

# Default command
CMD ["sh", "-c", "faceberg ${{CATALOG_URI}} serve --host 0.0.0.0 --port 7860"]
