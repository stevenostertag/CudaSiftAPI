# --- Stage 1: Downloader ---
# This stage uses the clean python:3.12-slim image, which we know can connect
# to PyPI, to download all the packages as wheel files.
FROM python:3.12-slim AS downloader

ARG http_proxy
ARG https_proxy

# Set them as environment variables for the duration of the build and for subsequent tools
ENV http_proxy=$http_proxy
ENV https_proxy=$https_proxy

WORKDIR /wheels

# Copy requirements file
COPY ./cusift_py/requirements.txt .

# Download all packages and their dependencies to the current directory
# This populates /wheels/ with all the .whl files we need.
RUN pip download --no-cache-dir -r requirements.txt


# --- Stage 2: Builder ---
# This stage is the same as before, building your C++/CUDA library.
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04 AS builder

ARG http_proxy
ARG https_proxy

# Set them as environment variables for the duration of the build and for subsequent tools
ENV http_proxy=$http_proxy
ENV https_proxy=$https_proxy

ENV DEBIAN_FRONTEND=noninteractive
ENV BUILD_DIR=/app/build

RUN echo 'Acquire::http::Proxy "'$http_proxy'";' > /etc/apt/apt.conf.d/01proxy
RUN echo 'Acquire::https::Proxy "'$https_proxy'";' >> /etc/apt/apt.conf.d/01proxy

RUN apt-get update && apt-get install -y --no-install-recommends cmake build-essential libtiff-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN rm -rf ${BUILD_DIR}
RUN cmake -S . -B ${BUILD_DIR} -DCUSIFT_BUILD_SHARED=ON -DCMAKE_BUILD_TYPE=Release
RUN cmake --build ${BUILD_DIR} --target cusift -j$(nproc)


# --- Stage 3: Final Image ---
# This is our final runtime image. It will install packages from the local
# wheel files instead of trying to connect to PyPI.
FROM nvidia/cuda:13.1.1-base-ubuntu24.04 AS app-runner

ARG http_proxy
ARG https_proxy

# Set them as environment variables for the duration of the build and for subsequent tools
ENV http_proxy=$http_proxy
ENV https_proxy=$https_proxy

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

RUN echo 'Acquire::http::Proxy "'$http_proxy'";' > /etc/apt/apt.conf.d/01proxy
RUN echo 'Acquire::https::Proxy "'$https_proxy'";' >> /etc/apt/apt.conf.d/01proxy

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Setup the virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application code and the compiled library
COPY ./cusift_py/ .
COPY --from=builder /app/build/libcusift.so /usr/local/lib/libcusift.so

# Copy the downloaded wheel files from the 'downloader' stage
COPY --from=downloader /wheels /wheels

# Install Python dependencies from the local wheel files.
# This does NOT connect to the internet, bypassing all SSL issues.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Clean up the wheel files after installation
RUN rm -rf /wheels

# Set the entrypoint
ENTRYPOINT ["python3", "cuda_optimize.py"]
