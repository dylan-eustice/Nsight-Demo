# Overview
The code in src/main.cu is a contrived data processing pipeline built to highlight NVIDIA's profiling tools and a couple of common pitfalls when getting started with CUDA programming. There are 4 different implementations of the same pipeline: they are all functionally identical, but differ in their relative runtime performance. The implementations are:
1. Baseline
2. Launch pipeline with >1 thread blocks
3. Process pipeline with CUDA streams to prevent unnecessary blocking
4. Use coalesced memory access ([relevant blog post](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels))

A graphical representation of the contrived pipeline:
![image](https://github.com/dylan-eustice/Nsight-Demo/assets/105389906/e4a33fb2-c55d-4164-a47a-cd34b8358350)

# Building and running
## Creating and running using Docker
A Dockerfile, along with bash scripts for building and running the Docker container are located in the `deploy` directory. To build and run the container, use:
```bash
./deploy/build-docker.sh
./deploy/run-docker.sh
```

## Building the executable
The application is built with CMake:
```bash
mkdir build && cd build
cmake ..
make
```

# Profiling tools
Run the application with `./nsight-demo`.

A script for generating an Nsight Systems report (.nsys-rep) and Nsight Compute reports (.ncu-rep) can be run with `./deploy/profile.sh`. The reports will be saved in the `./nsys-reports` directory.
