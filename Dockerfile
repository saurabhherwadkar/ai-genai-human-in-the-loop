FROM python:3.11-slim as builder
WORKDIR /app
RUN pip install --no-cache-dir poetry==1.8.4
COPY pyproject.toml poetry.lock* ./
RUN poetry export -f requirements.txt --without-hashes --output requirements.txt

FROM python:3.11-slim as runtime
WORKDIR /app
RUN useradd --create-home --shell /bin/bash appuser
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY config/ ./config/
RUN mkdir -p logs && chown -R appuser:appuser /app
USER appuser
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "human_in_the_loop.main:app", "--host", "0.0.0.0", "--port", "8000"]
