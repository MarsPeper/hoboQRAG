FROM python:3.12-slim

# Install minimal build tools if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pre-install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY . .

# Expose FastAPI port
EXPOSE 5000

# Set runtime settings
ENV TESTING=false
ENV HOST=0.0.0.0
ENV PORT=5000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
