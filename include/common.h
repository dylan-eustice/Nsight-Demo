#pragma once

#include <iostream>
#include <nvToolsExt.h>

#define ROWS 100000
#define COLS 8
#define NUM_ITERS 5

#define CUDA_CHECK(err) { \
  cudaError_t __cuer = err; \
  if (__cuer != cudaSuccess) { \
    std::cerr << "[CUDA Error] " << ::cudaGetErrorString(__cuer) \
              << " in " << __FILE__ << "::" << __LINE__ \
              << std::endl; \
    ::exit(1); \
  } \
}

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_NVTX_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id % num_colors; \
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_NVTX_RANGE nvtxRangePop();

float rand_float() {
  return 2 * (rand() / static_cast<float>(RAND_MAX)) - 1;
}

void generate_data(float *data, const size_t rows, const size_t cols) {
  PUSH_NVTX_RANGE("generate_data", 1);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      data[i + j*rows] = rand_float();
    }
  }
  POP_NVTX_RANGE;
}