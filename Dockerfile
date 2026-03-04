FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
COPY drawing_qa.py .
COPY README.md .

RUN pip install --no-cache-dir -e ".[server]"

# Create index directory
RUN mkdir -p /app/drawing_index

# Expose port
EXPOSE 8000

# Default command
CMD ["drawing-qa", "server", "--host", "0.0.0.0", "--port", "8000"]
