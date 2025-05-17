
FROM python:3.8-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory (should match where you COPY your app files)
WORKDIR /app

# Copy requirements and install
COPY Requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r Requirements.txt

# Copy all app files into the container
COPY . .

# Expose the port your Flask app will run on
EXPOSE 5000

# Default command to run Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
