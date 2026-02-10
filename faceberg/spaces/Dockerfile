FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apk update
RUN apk add git

RUN uv pip install --system --no-cache git+https://github.com/kszucs/faceberg.git@oauth

EXPOSE 7860

CMD sh -c 'faceberg hf://spaces/${SPACE_ID} serve --host 0.0.0.0 --port 7860'
