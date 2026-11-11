# Build context is the repo root (see docker-compose.yml) so the core package can be installed.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev libglib2.0-0 libgomp1 xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY core /app/core
RUN pip install --no-cache-dir "/app/core[render]"

COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

COPY server/app /app/server/app
ENV PYTHONPATH=/app/server

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
