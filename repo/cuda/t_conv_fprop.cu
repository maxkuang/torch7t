
#include <cuda_header.h>

//#define CUDA_SHARED_MEM_SIZE 1024-32
//#define CUDA_SHARED_MEM_SIZE 4*1024-18 // 16K/SM13
#define CUDA_SHARED_MEM_SIZE 12*1024-32 // 48K/SM20

template <bool swapkernel, int T_kernel_h, int T_kernel_w>
  __global__ void conv2generic(float *input, float *kernel, float *output,
                               int input_n, int input_h, int input_w,
                               int kernel_n, int kernel_h, int kernel_w,
                               int stride_h, int stride_w)
{
  // output dimensions
  int output_h = (input_h - kernel_h) / stride_h + 1;
  int output_w = (input_w - kernel_w) / stride_w + 1;

  // xcorr or conv
  int koffset = swapkernel ? kernel_w*kernel_h-1 : 0;

  // nb outputs
  int output_n = kernel_n / input_n;

  // generate offsets according to block/thread ids
  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step =blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y*gridDim.y;

  int oo_start = blockIdx.x;
  int oo_end = oo_start+1;

  int ii_start = (blockIdx.x / output_n) * input_n;
  int ii_end = ii_start + input_n;

  // nb threads, unique thread id
  int tid = blockDim.x*blockDim.y*threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  int nthreads = blockDim.x * blockDim.y * blockDim.z;

  // iterators
  int oo, ii, xx, yy, kx, ky, kk;

  // do the kernels fit in shared mem ?
  if (input_n*kernel_w*kernel_h <= CUDA_SHARED_MEM_SIZE) {

    // put the kernel in shared memory
    __shared__ float shared_kernel[CUDA_SHARED_MEM_SIZE];

    // first thread of each block does the copy
    for (kk = tid; kk < kernel_w*kernel_h*input_n; kk += nthreads) {
      shared_kernel[kk] = kernel[input_n*kernel_w*kernel_h*(oo_start % output_n) + kk];
    }
    __syncthreads();

    // templated kernel size
    if ((T_kernel_w > 0) && (T_kernel_h > 0)) {
      // unrolled convolution loop
      for(oo = oo_start; oo < oo_end; oo++) {
	for(ii = ii_start; ii < ii_end; ii++) {

	  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
	    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
              // Dot product in two dimensions... (between input image and the mask)
              float *input_p = input + ii*input_h*input_w + yy*stride_h*input_w + xx*stride_w;
              float *output_p = output + oo*output_h*output_w + yy*output_w + xx;
              float *kernel_p = shared_kernel + (ii % input_n)*kernel_w*kernel_h + koffset;
              float sum = 0;
              if (swapkernel) {
#pragma unroll
                for(ky = 0; ky < T_kernel_h; ky++) {
#pragma unroll
                  for(kx = 0; kx < T_kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p--);
                  }
                  input_p += input_w;
                }
              } else {
#pragma unroll
                for(ky = 0; ky < T_kernel_h; ky++) {
#pragma unroll
                  for(kx = 0; kx < T_kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p++);
                  }
                  input_p += input_w;
                }
              }
              *output_p += sum;
            }
          }
        }
      }
    }
  } else { // not enough shared mem for kernels, simply stream them
  }
}

// mb = 1
// input: nInputPlane,nInputRows,nInputCols
// kernel: nOutputPlane,nInputPlane,nKernelRows,nKernelCols
// output: nOutputPlane,nOutputRows,nOutputCols
void conv2Dmv_test1() {
  long nInputPlane(32), nInputRows(32), nInputCols(32); // no padding
  long nKernelRows(5), nKernelCols(5);
  long nOutputPlane(128), nOutputRows(28), nOutputCols(28); // stride = 1

  float *input = cuda_new(nInputPlane*nInputRows*nInputCols);
  float *kernel = cuda_new(nOutputPlane*nInputPlane*nKernelRows*nKernelCols);
  float *output = cuda_new(nOutputPlane*nOutputRows*nOutputCols);
  //  cudaDeviceSynchronize();

  dim3 blocks(nOutputPlane,1);
  dim3 threads(32,8);  

  //  clock_t t;
  //  t = clock();
  //  printf ("conv2Dmv_test1:conv2generic...");
  conv2generic <false, 5, 5> <<<blocks, threads>>> 
    (input, kernel, output,
     nInputPlane, nInputRows, nInputCols,
     nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
     1, 1); 
  //  cudaDeviceSynchronize();  
  //  t = clock() - t;
  //  printf("in %f secs.\n",((float)t)/CLOCKS_PER_SEC);  

  cuda_free(input);
  cuda_free(kernel);
  cuda_free(output);
  
  cudaDeviceSynchronize();  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in conv2Dmv_test1: %s\n", cudaGetErrorString(err));
  }
}

// mb = 128
// input: nBatch,nInputPlane,nInputRows,nInputCols
// kernel: nOutputPlane,nInputPlane,nKernelRows,nKernelCols
// output: nBatch,nOutputPlane,nOutputRows,nOutputCols
void conv2Dmm_test2() {
  long nBatch(128);
  long nInputPlane(32), nInputRows(32), nInputCols(32); // no padding
  long nKernelRows(5), nKernelCols(5);
  long nOutputPlane(128), nOutputRows(28), nOutputCols(28); // stride = 1

  float *input = cuda_new(nBatch*nInputPlane*nInputRows*nInputCols);
  float *kernel = cuda_new(nOutputPlane*nInputPlane*nKernelRows*nKernelCols);
  float *output = cuda_new(nBatch*nOutputPlane*nOutputRows*nOutputCols);

  dim3 blocks(nOutputPlane*nBatch,1);
  dim3 threads(32,8);  

  conv2generic <false, 5, 5> <<<blocks, threads>>> 
    (input, kernel, output,
     nInputPlane, nInputRows, nInputCols,
     nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
     1, 1); 

  cuda_free(input);
  cuda_free(kernel);
  cuda_free(output);

  cudaDeviceSynchronize();  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in conv2Dmm_test2: %s\n", cudaGetErrorString(err));
  }
}

int main() {
  cuda_init();
  clock_t t;    
  for (int i = 0; i < 10; ++i) {
    tic(&t,"conv2Dmv_test1...");
    conv2Dmv_test1();
    toc(&t, "done");

    tic(&t, "conv2Dmm_test2...");
    conv2Dmm_test2();
    toc(&t, "done");
  }
  return 0;
}
