# Using NVIDIA PyTorch base for CUDA support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Setting the AIR environment variables
ENV AIR_HOME=AIR
ENV DEBIAN_FRONTEND=noninteractive

# Installing system-level dependencies for OpenCV and Linux stability
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    && ln -s /usr/bin/python3.11 /usr/local/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

# Setting the working directory
WORKDIR /AIR

# Copying only requirements first to leverage Docker cache
COPY requirements.txt .

# Installing Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copying the rest of the AIR project files
COPY . .

# Default command
CMD ["bash"]