# Base image
FROM nvidia/cudagl:10.1-devel-ubuntu16.04

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    fontconfig \
    unzip && \
    rm -rf /var/lib/apt/lists/*
    
# Install conda
RUN curl -L -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Conda environment
RUN conda create -n habitat python=3.9 cmake=3.14.0

# Setup habitat-sim
RUN git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
RUN /bin/bash -c ". activate habitat; cd habitat-sim; pip install -r requirements.txt; python setup.py install --headless"

# # Install challenge specific habitat-lab
# RUN git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
# RUN /bin/bash -c ". activate habitat; cd habitat-lab; pip install -e habitat-lab/"

# Silence habitat-sim logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"

# COPY dir 
# COPY . /hm3d_collector
# RUN /bin/bash -c ". activate habitat; cd /hm3d_collector; pip install -r requirements.txt; pip install ."

RUN git clone --branch stable https://github.com/EnriqueSolarte/hm3d_data_collector.git
RUN /bin/bash -c ". activate habitat; cd hm3d_data_collector; pip install -r requirements.txt; pip install ."

ARG USER=nobody
ARG USERNAME=$USER-docker
ARG USER_UID=1002
ARG USER_GID=$USER_UID

# Create a non-root user
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  # Add sudo support for the non-root user
  && apt-get update \
  && apt-get install -y --no-install-recommends sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && rm -rf /var/lib/apt/lists/*

COPY setup.sh /etc/profile.d/init.sh
RUN chmod +x /etc/profile.d/init.sh
  
USER $USERNAME
# Set the default shell to bash
ENV TERM=xterm-256color
ENV USER=$USERNAME
CMD ["bash", "-l"]