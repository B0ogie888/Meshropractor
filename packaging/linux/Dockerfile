FROM python:3.12-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libegl1 libopengl0 libglx-mesa0 libegl-mesa0 libgl1-mesa-dri \
    libglib2.0-0 libgomp1 libx11-6 libxext6 \
    libxrender1 libxfixes3 libxrandr2 libxcursor1 libxi6 libxkbcommon0 \
    libxkbcommon-x11-0 libxcb1 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-randr0 libxcb-render0 libxcb-render-util0 \
    libxcb-shape0 libxcb-shm0 libxcb-sync1 libxcb-xfixes0 libxcb-xkb1 \
    libfontconfig1 libfreetype6 libdbus-1-3 libnss3 libasound2 fonts-dejavu-core \
    xvfb xauth xdotool dpkg-dev file binutils xz-utils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements.txt requirements-dev.txt /build/
RUN python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements-dev.txt
CMD ["bash", "/workspace/packaging/linux/build.sh"]
