FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

# Install some basic utilities
RUN mkdir -p /opt/ml/code \ 
    && apt-get update && apt-get install -y \
    software-properties-common \
    && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    libgl1-mesa-dev \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip \
    && pip install --upgrade pip

WORKDIR /opt/ml/code
