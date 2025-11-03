# Use slim Python 3.11 base image
FROM python:3.11-slim

# # Update system packages to fix vulnerabilities
# USER root
# # trunk-ignore(hadolint/DL3018)
# RUN apk update && apk upgrade && apk add --no-cache bash

# Set working directory
WORKDIR /app

# Copy and install Python dependencies

# Install build dependencies for PEP 517/518
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Normalize line endings in entrypoint.sh (Windows to Unix)
RUN sed -i 's/\r$//' entrypoint.sh

# Create logs directory inside container
# Make entrypoint script executable
RUN mkdir -p logs && chmod +x entrypoint.sh

EXPOSE 8000

# Add a HEALTHCHECK instruction
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8000/health || exit 1

# Use shell to run the entrypoint script
ENTRYPOINT ["bash", "./entrypoint.sh"]