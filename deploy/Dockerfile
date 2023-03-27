FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu20.04 AS devel

# Install nice-to-haves
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      sudo git vim build-essential libtool autoconf unzip make wget curl \
      cmake iputils-ping python3-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install Nsight Systems & Nsight Compute
ARG NSYS_VER="2023.1.1"
ARG NCU_VER="2023.1.0"
ARG NV_URL="https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64"
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-transport-https ca-certificates gnupg wget && \
    rm -rf /var/lib/apt/lists/* && \
    wget -qO - "${NV_URL}/nvidia.pub" | apt-key add - && \
    echo "deb ${NV_URL}/ /" >> /etc/apt/sources.list.d/nsight.list
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        nsight-systems-${NSYS_VER} \
        nsight-compute-${NCU_VER} && \
    rm -rf /var/lib/apt/lists/* 
RUN ln -s /opt/nvidia/nsight-compute/2023.1.0/ncu /usr/local/bin/ncu

# Install Python packages
RUN pip3 install scipy numpy setuptools nvtx

# Install CMake
ARG CMAKE_VER="3.26.0-rc4"
ARG CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-linux-x86_64.sh"
RUN mkdir -p /var/tmp && \
    wget -q -nc --no-check-certificate -P /var/tmp ${CMAKE_URL} && \
    mkdir -p /usr/local && \
    /bin/sh /var/tmp/cmake-${CMAKE_VER}-linux-x86_64.sh --prefix=/usr/local --skip-license && \
    rm -rf /var/tmp/cmake-${CMAKE_VER}-linux-x86_64.sh
ENV PATH=/usr/local/bin:$PATH

# Create 'docker' user
RUN addgroup --gid 1000 docker && \
    adduser --uid 1000 \
            --ingroup docker \
            --home /home/docker \
            --shell /bin/bash \
            --disabled-password \
            --gecos "" docker
RUN echo "docker:docker" | chpasswd && adduser docker sudo

# Install fixuid
ARG FIXUID_URL="https://github.com/boxboat/fixuid/releases/download/v0.5.1/fixuid-0.5.1-linux-amd64.tar.gz"
RUN USER=docker && \
    GROUP=docker && \
    curl -SsL ${FIXUID_URL} | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml && \
    echo "docker ALL = (root) NOPASSWD: ALL" >> /etc/sudoers

# Set entrypoint
USER docker:docker
WORKDIR /home/docker