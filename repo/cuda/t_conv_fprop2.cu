
#include <cuda_header.h>

// manual control of shared mem and register 
//
// input: mb,ic,ih,iw (minibatch,input_c,input_h,input_w)
// kernel: oc,ic,kh,kw (output_c,input_c,kernel_h,kernel_w)
// output: mb,oc,oh,ow (minibatch,output_c,output_h,output_w)
// kernel_n = output_c * input_c
//
// (if) not swapkernel
//   output[mb,oc,oh,ow] = sum_(ic,kh,kw) 
//           kernel[oc,ic,kh,kw] * input[mb,ic,oh+kh,ow+kw]
//
//   gridDim.x = output_c
//   gridDim.y = minibatch
//   blockDim.x = partition output_w by Bow: 32
//   blockDim.y = partition opuput_h by Boh: 8
//
//   l1 cache for input: TODO CudaL1Perferred
//   shared mem for kernel and output: 2KB/block
//     Ckw = #kw to cache: 16,8
//     Ckh = #kh to cache: 16,8
//     Cic = #ic to cache: 1,4
//     assume kernel_h <= Ckh, kernel_w <= Ckw
//   #register: 32/thread?
//
template <bool swapkernel, int Bow, int Boh, int Cic, int Ckh, int Ckw>
  __global__ void _conv2generic(float *input, float *kernel, float *output,
				int input_c, int input_h, int input_w,
				int kernel_n, int kernel_h, int kernel_w,
				int stride_h, int stride_w) {

  // process Boh*Bow output pixels for every Cic input channels
  __shared__ float sh_kernel[Cic][Ckh][Ckw]; // 1*16*16*4Bytes = 1KB
  __shared__ float sh_output[Boh][Bow]; // 32*8*4Bytes = 1KB
  
  // output dimensions
  const int output_h = (input_h - kernel_h) / stride_h + 1;
  const int output_w = (input_w - kernel_w) / stride_w + 1;
  const int output_c = kernel_n / input_c; 

  const int ow_start = threadIdx.x;
  const int ow_end = output_w;
  const int ow_step = Bow;
  
  const int oh_start = threadIdx.y;
  const int oh_end = output_h;
  const int oh_step = Boh;

  const int oc = blockIdx.x;
  const int mb = blockIdx.y;

  input  += mb*input_c*input_h*input_w; // [mb]*
  kernel += oc*input_c*kernel_h*kernel_w; // [oc]*
  output += mb*output_c*output_h*output_w + oc*output_h*output_w; //[mb][oc]*

  int ic,kh,kw,oh,ow;
  for (oh = oh_start; oh < oh_end; oh+=oh_step) {
    for (ow = ow_start; ow < ow_end; ow+=ow_step) {
      // smem: init output kernel 
      sh_output[oh%oh_step][ow%ow_step] = 0;
      // sh_output[threadIdx.y][threadIdx.x] = 0;
      __syncthreads();

      for (ic = 0; ic < input_c; ic+=Cic) {
	// smem: read kernel[oc][ic][*][*]
	for (kh = threadIdx.y; kh < kernel_h; kh+=Boh) {
	  for (kw = threadIdx.x; kw < kernel_w; kw+=Bow) {
	    sh_kernel[ic%Cic][kh][kw] = 
	      kernel[ic*kernel_h*kernel_w + kh*kernel_w + kw];
	  }
	}
	__syncthreads();

	// conv
	float csum = 0;
	float *_input = 
	  input + ic*input_h*input_w + oh*stride_h*input_w + ow*stride_w;
	for (kh = 0; kh < kernel_h; ++kh) {
	  for (kw = 0; kw < kernel_w; ++kw) {
	    csum += sh_kernel[ic%Cic][kh][kw] * _input[kw];
	  }
	  _input += input_w;
	}
	sh_output[threadIdx.y][threadIdx.x] += csum;
      }

      // write output
      output[oh*output_w+ow] = sh_output[threadIdx.y][threadIdx.x];
      __syncthreads();      
    }
  }
}


// mb = 1
// input: nInputPlane,nInputRows,nInputCols
// kernel: nOutputPlane,nInputPlane,nKernelRows,nKernelCols
// output: nOutputPlane,nOutputRows,nOutputCols
void _conv2Dmv_test1() {
  long nInputPlane(32), nInputRows(32), nInputCols(32); // no padding
  long nKernelRows(5), nKernelCols(5);
  long nOutputPlane(128), nOutputRows(28), nOutputCols(28); // stride = 1

  float *input = cuda_new(nInputPlane*nInputRows*nInputCols);
  float *kernel = cuda_new(nOutputPlane*nInputPlane*nKernelRows*nKernelCols);
  float *output = cuda_new(nOutputPlane*nOutputRows*nOutputCols);

  dim3 blocks(nOutputPlane,1);
  dim3 threads(32,8);  

  _conv2generic <false, 32, 8, 8, 8, 4> <<<blocks, threads>>> 
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
    printf("error in _conv2Dmv_test1: %s\n", cudaGetErrorString(err));
  }
}

// mb = 128
// input: nBatch,nInputPlane,nInputRows,nInputCols
// kernel: nOutputPlane,nInputPlane,nKernelRows,nKernelCols
// output: nBatch,nOutputPlane,nOutputRows,nOutputCols
void _conv2Dmm_test2() {
  long nBatch(128);
  long nInputPlane(32), nInputRows(32), nInputCols(32); // no padding
  long nKernelRows(5), nKernelCols(5);
  long nOutputPlane(128), nOutputRows(28), nOutputCols(28); // stride = 1

  float *input = cuda_new(nBatch*nInputPlane*nInputRows*nInputCols);
  float *kernel = cuda_new(nOutputPlane*nInputPlane*nKernelRows*nKernelCols);
  float *output = cuda_new(nBatch*nOutputPlane*nOutputRows*nOutputCols);

  dim3 blocks(nOutputPlane,nBatch);
  dim3 threads(32,8);  

  _conv2generic <false, 32, 8, 8, 8, 4> <<<blocks, threads>>> 
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
    printf("error in _conv2Dmm_test2: %s\n", cudaGetErrorString(err));
  }
}


int main() {
  cuda_init();
  clock_t t;    
  for (int i = 0; i < 10; ++i) {
    tic(&t,"_conv2Dmv_test1...");
    _conv2Dmv_test1();
    toc(&t, "done");

    tic(&t, "_conv2Dmm_test2...");
    _conv2Dmm_test2();
    toc(&t, "done");
  }
  return 0;
}
