# Creating and running using Docker
A Dockerfile, along with bash scripts for building and running the Docker container are located in the `deploy` directory. To build and run the container, use:
```bash
./deploy/build-docker.sh
./deploy/run-docker.sh
```

# Building executable
The application is built with CMake:
```bash
mkdir build && cd build
cmake ..
make
```

# Running and creating profile reports
Run the application with `./nsight-demo`.

A script for generating an Nsight Systems report (.nsys-rep) and Nsight Compute reports (.ncu-rep) can be run with `./deploy/profile.sh`. The reports will be saved in the `./nsys-reports` directory.