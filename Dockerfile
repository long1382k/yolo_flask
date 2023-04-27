
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libssl-dev \
    libffi-dev \
    python3-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the Flask app files
COPY . .

# Expose the Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "my_api.py"]
