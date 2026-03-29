FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=3).raise_for_status()"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
