# spillway-forecast worker: `spillway watch` (refresh every hour, retrain monthly).
# The trained model is NOT baked in: models/ and reports/ are mounted (see docker-compose.yml).
# In production mount a persistent volume there and run `spillway retrain --force` once.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra lake --extra warehouse

ENV PATH="/app/.venv/bin:$PATH" \
    SPILLWAY_DATA_DIR=/data \
    MLFLOW_TRACKING_URI=sqlite:////data/mlflow.db
VOLUME ["/data"]
RUN useradd --system --create-home worker && mkdir -p /data /app/models /app/reports && chown -R worker /data /app/models /app/reports
USER worker
ENTRYPOINT ["spillway"]
CMD ["watch", "--every", "60"]
