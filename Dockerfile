FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py cora_data.npz simple_gcn_cora.onnx simple_gcn_cora.onnx.data ./
COPY static/ ./static/

# Hugging Face Spaces expects 7860; Cloud Run and most PaaS set $PORT.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
