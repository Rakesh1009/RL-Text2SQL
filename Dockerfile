FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install git and other system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy requirements file
COPY requirements.txt .

# Install dependencies using uv in a virtual environment
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache-dir -r requirements.txt

# Ensure the virtual environment is used for all subsequent commands
ENV PATH="/opt/venv/bin:$PATH"

# Copy the rest of the source code
COPY . .

# Default command
CMD ["python", "run_train.py"]
