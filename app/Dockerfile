
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

# Set working directory (matches your local `devpro2/app` folder)
WORKDIR /app

# Copy requirements and install
COPY Requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r Requirements.txt

# Copy the entire app folder (including Jupyter notebook and CSV)
COPY . .

# Expose Jupyter port (optional: only needed if running Jupyter)
EXPOSE 8888

# Default command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]