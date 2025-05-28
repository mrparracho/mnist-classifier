FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    netcat-openbsd \
    iputils-ping \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create checkpoints directory
RUN mkdir -p /app/checkpoints

# Copy wait-for-it script
COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# Copy the model package
COPY . /app/

# Install the model package in development mode
WORKDIR /app

# Add the app directory to PYTHONPATH
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 8000

# Command to run the application with wait-for-it
CMD ["/wait-for-it.sh", "db:5432", "--", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
