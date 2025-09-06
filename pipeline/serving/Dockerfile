FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir "transformers" "torch" "fastapi" "uvicorn" "accelerate"
COPY serve.py /app/serve.py
ENV MODEL_DIR=/model
EXPOSE 8008
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8008"]