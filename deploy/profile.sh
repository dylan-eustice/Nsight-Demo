#!/usr/bin/bash

# Make directory if it doesn't exist
if [ ! -d "$(dirname $0)/../nsys-reports" ]; then
  echo "Creating $(dirname $0)/../nsys-reports"
  mkdir "$(dirname $0)/../nsys-reports"
fi

# Nsight System profile
nsys profile \
  -o $(dirname $0)/../nsys-reports/nsight-demo \
  -f true \
  $(dirname $0)/../build/nsight-demo

# Nsight Compute profile un-coalesced operations kernel
sudo ncu \
  -f -o $(dirname $0)/../nsys-reports/operations_fullgrid \
  --kernel-name operations_fullgrid \
  --launch-skip 2 \
  --launch-count 1 \
  --set full \
  $(dirname $0)/../build/nsight-demo

# Nsight Compute profile coalesced operations kernel
sudo ncu \
  -f -o $(dirname $0)/../nsys-reports/operations_coalesced \
  --kernel-name operations_coalesced \
  --launch-skip 2 \
  --launch-count 1 \
  --set full \
  $(dirname $0)/../build/nsight-demo