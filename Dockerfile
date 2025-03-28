# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
# ffmpeg - http://ffmpeg.org/download.html
#
# From https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu
#
# https://hub.docker.com/r/jrottenberg/ffmpeg/
#
#
FROM runpod/base:0.6.2-cuda12.2.0 as base

ENV PKG_CONFIG_PATH=/usr/lib/arm-linux-gnueabihf/pkgconfig/:/usr/local/lib/pkgconfig/
# Update base and install build tools
RUN apt-get update
# Install ffmpeg libraries
RUN apt-get install -y yasm pkg-config nasm unzip

# --- Optional: System dependencies ---
COPY ./runpod-setup.sh /setup.sh
RUN /bin/bash /setup.sh && \
    rm /setup.sh

# Python dependencies
COPY requirements.txt /requirements.txt
RUN PIP_REQUIRE_HASHES= python3.11 -m pip install --upgrade pip && \
    PIP_REQUIRE_HASHES= python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt
    

ADD . .

# NOTE: The base image comes with multiple Python versions pre-installed.
#       It is reccommended to specify the version of Python when running your code.

# Add src files (Worker Template)
CMD python3.11 -u /runpod_handler.py