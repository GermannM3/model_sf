FROM python:3.9-slim as builder

WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

COPY . .

FROM python:3.9-slim
WORKDIR /app

COPY --from=builder /app /app

ENV PYTHONPATH=/app
CMD ["python", "-m", "src.main"] 