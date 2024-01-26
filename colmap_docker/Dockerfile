FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

LABEL maintainer="Lacie"

ARG DEBIAN_FRONTEND=non-interactive
ARG CMAKE_CUDA_ARCHITECTURES=89

# install general dependencies

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    vim

RUN apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

RUN apt-get update && apt-get install -y \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev

RUN mkdir /tools

# install colmap

WORKDIR /tools
RUN git clone https://github.com/colmap/colmap.git
WORKDIR /tools/colmap
RUN mkdir build
WORKDIR /tools/colmap/build
RUN cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=89
RUN ninja -j4 && ninja install

# remove unneeded files
RUN rm -rf /tools

# RUN mkdir /home
RUN chmod -R 777 /home
ENV HOME /home
WORKDIR /home