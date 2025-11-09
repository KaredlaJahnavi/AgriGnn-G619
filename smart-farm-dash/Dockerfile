FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for healthcheck
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Install torch via pip (let pip pick CPU if no CUDA GPU is present)
RUN pip install --upgrade pip \
    && pip install torch --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py yield_model.py dataset.csv ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
