FROM python:3.11-slim

WORKDIR /app

COPY infra/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY src ./src

ENV PORT=8080

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]
