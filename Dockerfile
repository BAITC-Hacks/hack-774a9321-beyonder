# Запуск одной командой: docker compose up --build  →  http://localhost:8000
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Каталог, предпосчитанные эмбеддинги и интерфейс — внутри образа: сеть и ключ для подбора не нужны.
COPY app ./app
COPY static ./static
COPY data ./data
COPY scripts ./scripts
COPY tests ./tests

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/meta', timeout=2)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
