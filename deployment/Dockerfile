# Dockerfile for Jetson Nano Deployment
# Supports: GStreamer, Ultralytics, Pymavlink, OpenCV (built from source with GStreamer support)
# Optimization: TensorRT on JetPack 4.6 (Jetson Nano)

FROM ultralytics/ultralytics:latest-jetson-jetpack4

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH" \
    CMAKE_ARGS="-DWITH_GSTREAMER=ON" \
    ENABLE_CONTRIB=0 

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Step: Build OpenCV from source with GStreamer support
# This layer is cached: it only runs ONCE. After the first build, it stays in the image.
# We do NOT remove the build directory here as requested, allowing you to inspect it.
# NOTE: Building this image persists it on your system. Subsequent 'docker run' commands take <1s.
RUN uv pip uninstall --system opencv-python opencv-contrib-python || true && \
    mkdir -p /tmp/opencv_build && cd /tmp/opencv_build && \
    git clone --recursive --depth 1 https://github.com/opencv/opencv-python.git && \
    cd opencv-python && \
    # Build the wheel (this takes ~2 hours on Jetson Nano, but ONLY during the first build)
    python3 -m pip wheel . --verbose && \
    # Install the generated wheel
    python3 -m pip install opencv_python*.whl
# Note: rm -rf /tmp/opencv_build was removed per user request to keep build materials.

# Install other Python dependencies using uv (Cached layer)
RUN uv pip install --system \
    ultralytics \
    pymavlink \
    numpy==1.23.5 \
    pyserial

WORKDIR /app

COPY . /app

CMD ["python3", "test.py"]
