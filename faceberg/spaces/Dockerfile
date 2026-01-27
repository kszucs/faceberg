FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install faceberg from GitHub
RUN uv pip install --system --no-cache git+https://github.com/{github_repo}.git

# Expose HuggingFace Spaces default port
EXPOSE 7860

# Default command
CMD ["sh", "-c", "faceberg ${{CATALOG_URI}} serve --host 0.0.0.0 --port 7860"]
