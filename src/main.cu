#include "common.h"

//----- Normalization functions ------------------------------------------
__global__
void normalize_baseline(float *data, const size_t rows, const size_t cols) {
  for (size_t i = threadIdx.x; i < rows; i += blockDim.x) {
    // Compute the sum of the row
    float tot = 0;
    for (size_t j = 0; j < cols; j++) {
      tot += data[i + j * rows];
    }
    // Normalize the row
    for (size_t j = 0; j < cols; j++) {
      data[i + j * rows] /= tot;
    }
  }
}

__global__
void normalize_fullgrid(float *data, const size_t rows, const size_t cols) {
  //* Step through all rows, with the step size being all threads in the grid
  const size_t ix_thread = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = ix_thread; i < rows; i += gridDim.x * blockDim.x) {
    // Compute the sum of the row
    float tot = 0;
    for (size_t j = 0; j < cols; j++) {
      tot += data[i + j * rows];
    }
    // Normalize the row
    for (size_t j = 0; j < cols; j++) {
      data[i + j * rows] /= tot;
    }
  }
}

//----- Operation functions ----------------------------------------------
__global__
void operations_baseline(float *out,
                         const float *in,
                         const size_t rows,
                         const size_t cols,
                         const bool is_A) {
  const size_t tot_threads = blockDim.x * blockDim.y;
  const size_t ix_thread = threadIdx.y + threadIdx.x * blockDim.y;

  for (size_t ix = ix_thread; ix < rows * cols; ix += tot_threads) {
    out[ix] = is_A ? sinf(in[ix]) : cosf(in[ix]);
  }
}

__global__
void operations_fullgrid(float *out,
                         const float *in,
                         const size_t rows,
                         const size_t cols,
                         const bool is_A) {
  const size_t threads_per_block = blockDim.x * blockDim.y;
  const size_t tot_threads = threads_per_block * gridDim.x * gridDim.y;
  const size_t ix_block = blockIdx.y + blockIdx.x * gridDim.y;
  const size_t ix_thread = threadIdx.y + threadIdx.x * blockDim.y + ix_block * threads_per_block;

  //* Step through all rows, with the step size being all threads in the grid
  for (size_t ix = ix_thread; ix < rows * cols; ix += tot_threads) {
    out[ix] = is_A ? sinf(in[ix]) : cosf(in[ix]);
  }
}

__global__
void operations_coalesced(float *out,
                          const float *in,
                          const size_t rows,
                          const size_t cols,
                          const bool is_A) {
  const size_t threads_per_block = blockDim.x * blockDim.y;
  const size_t tot_threads = threads_per_block * gridDim.x * gridDim.y;
  const size_t ix_block = blockIdx.x + blockIdx.y * gridDim.x;

  //* Fix memory access pattern so that the reads/writes are coalesced
  const size_t ix_thread = threadIdx.x + threadIdx.y * blockDim.x + ix_block * threads_per_block;

  for (size_t ix = ix_thread; ix < rows * cols; ix += tot_threads) {
    out[ix] = is_A ? sinf(in[ix]) : cosf(in[ix]);
  }
}

//----- Pipeline functions -----------------------------------------------
void pipeline_baseline(float *output_A,
                       float *output_B,
                       float *output_C,
                       float *input) {

  // Do normalization
  PUSH_NVTX_RANGE("Normalize", 1);
  normalize_baseline<<<1, 1024>>>(input, ROWS, COLS);
  POP_NVTX_RANGE;

  // Copy normalized values to output array
  cudaMemcpy(output_C,
             input,
             ROWS*COLS*sizeof(float),
             cudaMemcpyDeviceToDevice);

  // Do operations
  dim3 grid_size(1, 1);
  dim3 block_size(32, 32);

  PUSH_NVTX_RANGE("Operation A", 2);
  operations_baseline<<<grid_size, block_size>>>(output_A, input, ROWS, COLS, true);
  POP_NVTX_RANGE;

  PUSH_NVTX_RANGE("Operation B", 3);
  operations_baseline<<<grid_size, block_size>>>(output_B, input, ROWS, COLS, false);
  POP_NVTX_RANGE;
}

void pipeline_fullgrid(float *output_A,
                       float *output_B,
                       float *output_C,
                       float *input) {

  // Do normalization
  PUSH_NVTX_RANGE("Normalize", 1);
  normalize_fullgrid<<<10, 1024>>>(input, ROWS, COLS);
  POP_NVTX_RANGE;

  // Copy normalized values to output array
  cudaMemcpy(output_C,
             input,
             ROWS*COLS*sizeof(float),
             cudaMemcpyDeviceToDevice);

  // Do operations
  dim3 grid_size(10, 1); //* utilize 10 thread-blocks
  dim3 block_size(32, 32);

  PUSH_NVTX_RANGE("Operation A", 2);
  operations_fullgrid<<<grid_size, block_size>>>(output_A, input, ROWS, COLS, true);
  POP_NVTX_RANGE;

  PUSH_NVTX_RANGE("Operation B", 3);
  operations_fullgrid<<<grid_size, block_size>>>(output_B, input, ROWS, COLS, false);
  POP_NVTX_RANGE;
}

void pipeline_streams(float *output_A,
                      float *output_B,
                      float *output_C,
                      float *input,
                      cudaStream_t &norm_stream,
                      cudaStream_t &opA_stream,
                      cudaStream_t &opB_stream) {

  // Do normalization
  PUSH_NVTX_RANGE("Normalize", 1);
  normalize_fullgrid<<<10, 1024, 0, norm_stream>>>(input, ROWS, COLS);
  cudaStreamSynchronize(norm_stream);
  POP_NVTX_RANGE;

  // Copy normalized values to output array
  //* Use non-blocking call to copy data
  cudaMemcpyAsync(output_C,
                  input,
                  ROWS*COLS*sizeof(float),
                  cudaMemcpyDeviceToDevice,
                  norm_stream);

  // Do operations
  //* Process each operation in a different stream so they run concurrently
  dim3 grid_size(10, 1);
  dim3 block_size(32, 32);

  PUSH_NVTX_RANGE("Operation A", 2);
  operations_fullgrid<<<grid_size, block_size, 0, opA_stream>>>(output_A, input, ROWS, COLS, true);
  POP_NVTX_RANGE;

  PUSH_NVTX_RANGE("Operation B", 3);
  operations_fullgrid<<<grid_size, block_size, 0, opB_stream>>>(output_B, input, ROWS, COLS, false);
  POP_NVTX_RANGE;
}

void pipeline_coalesced(float *output_A,
                        float *output_B,
                        float *output_C,
                        float *input,
                        cudaStream_t &norm_stream,
                        cudaStream_t &opA_stream,
                        cudaStream_t &opB_stream) {

  // Do normalization
  PUSH_NVTX_RANGE("Normalize", 1);
  normalize_fullgrid<<<10, 1024, 0, norm_stream>>>(input, ROWS, COLS);
  cudaStreamSynchronize(norm_stream);
  POP_NVTX_RANGE;

  // Copy normalized values to output array
  cudaMemcpyAsync(output_C,
                  input,
                  ROWS*COLS*sizeof(float),
                  cudaMemcpyDeviceToDevice,
                  norm_stream);

  // Do operations
  dim3 grid_size(10, 1);
  dim3 block_size(32, 32);

  PUSH_NVTX_RANGE("Operation A", 2);
  operations_coalesced<<<grid_size, block_size, 0, opA_stream>>>(output_A, input, ROWS, COLS, true);
  POP_NVTX_RANGE;

  PUSH_NVTX_RANGE("Operation B", 3);
  operations_coalesced<<<grid_size, block_size, 0, opB_stream>>>(output_B, input, ROWS, COLS, false);
  POP_NVTX_RANGE;
}

//----- Main function ----------------------------------------------------
int main(int argc, char **argv) {
  char nvtx_str[1024];

  // Initialize CUDA streams
  cudaStream_t norm_stream, opA_stream, opB_stream;
  CUDA_CHECK( cudaStreamCreateWithFlags(&norm_stream, cudaStreamNonBlocking) );
  CUDA_CHECK( cudaStreamCreateWithFlags(&opA_stream, cudaStreamNonBlocking) );
  CUDA_CHECK( cudaStreamCreateWithFlags(&opB_stream, cudaStreamNonBlocking) );

  // Generate data
  float cpu_input[ROWS*COLS];
  generate_data(cpu_input, ROWS, COLS);

  // Allocate and initialize device memory
  float *input, *output_A, *output_B, *output_C;
  const size_t nbytes = ROWS * COLS * sizeof(float);
  CUDA_CHECK( cudaMalloc((void **)&input, nbytes) );
  CUDA_CHECK( cudaMalloc((void **)&output_A, nbytes) );
  CUDA_CHECK( cudaMalloc((void **)&output_B, nbytes) );
  CUDA_CHECK( cudaMalloc((void **)&output_C, nbytes) );
  CUDA_CHECK( cudaMemcpy(input, cpu_input, nbytes, cudaMemcpyHostToDevice) );

  for (size_t iter = 0; iter < NUM_ITERS; iter++) {

    // Baseline pipeline
    sprintf(nvtx_str, "Baseline Test %lu", iter);
    PUSH_NVTX_RANGE(nvtx_str, 4);
    pipeline_baseline(output_A, output_B, output_C, input);
    cudaDeviceSynchronize();
    CUDA_CHECK( cudaGetLastError() );
    POP_NVTX_RANGE;

    // Full-grid pipeline
    sprintf(nvtx_str, "Full-grid Test %lu", iter);
    PUSH_NVTX_RANGE(nvtx_str, 5);
    pipeline_fullgrid(output_A, output_B, output_C, input);
    cudaDeviceSynchronize();
    CUDA_CHECK( cudaGetLastError() );
    POP_NVTX_RANGE;

    // Streams pipeline
    sprintf(nvtx_str, "Streams Test %lu", iter);
    PUSH_NVTX_RANGE(nvtx_str, 6);
    pipeline_streams(output_A, output_B, output_C, input, norm_stream, opA_stream, opB_stream);
    cudaDeviceSynchronize();
    CUDA_CHECK( cudaGetLastError() );
    POP_NVTX_RANGE;

    // Coalesced pipeline
    sprintf(nvtx_str, "Coalesced Test %lu", iter);
    PUSH_NVTX_RANGE(nvtx_str, 7);
    pipeline_coalesced(output_A, output_B, output_C, input, norm_stream, opA_stream, opB_stream);
    cudaDeviceSynchronize();
    CUDA_CHECK( cudaGetLastError() );
    POP_NVTX_RANGE;
  }

  // Free memory
  CUDA_CHECK( cudaFree(input) );
  CUDA_CHECK( cudaFree(output_A) );
  CUDA_CHECK( cudaFree(output_B) );
  CUDA_CHECK( cudaFree(output_C) );

  // Clean up CUDA streams
  CUDA_CHECK( cudaStreamDestroy(norm_stream) );
  CUDA_CHECK( cudaStreamDestroy(opA_stream) );
  CUDA_CHECK( cudaStreamDestroy(opB_stream) );
}