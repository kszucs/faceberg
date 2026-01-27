FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv pip install --system --no-cache \
    https://github.com/kszucs/faceberg/archive/refs/heads/main.tar.gz

EXPOSE 7860

CMD sh -c 'faceberg hf://spaces/${{SPACE_ID}} serve --host 0.0.0.0 --port 7860'
