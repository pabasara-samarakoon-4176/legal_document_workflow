FROM python:3.10-slim

WORKDIR /app
RUN pip install --no-cache-dir torch transformers fastapi uvicorn

COPY serve.py .

ENV AIP_STORAGE_URI=/model
EXPOSE 8080

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
