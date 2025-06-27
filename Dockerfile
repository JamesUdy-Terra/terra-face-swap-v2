# Use specific Python version
FROM python:3.9.6-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ROOP_HEADLESS=1

# Set working directory inside the container
WORKDIR /app

# Install OS-level dependencies (for ONNX, image processing, etc.)
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose port for GCP (Cloud Run expects 8080)
EXPOSE 8080

# Start the app (choose based on your framework)
# 🔄 If FastAPI:
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

# 🔄 If Flask or plain Python script:
#CMD ["python", "app.py"]
