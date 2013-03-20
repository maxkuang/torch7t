
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
// partition output_w by Bow blocks
//   gridDim.x = output_c*DIVUP(output_w/Bow)
// partition output_h by Boh blocks
//   gridDim.y = minibatch*DIVUP(output_h/Boh)
// each block deals with Bow*Boh output locations
//   blockDim.x = Bow
//   blockDim.y = Boh
//
//   l1 cache for input: TODO CudaL1Perferred
//   shared mem for kernel and output: 2KB/block
//     Ckw = #kw to cache: 16,8
//     Ckh = #kh to cache: 16,8
//     Cic = #ic to cache: 1,4
//   #register: 32/thread?
//
//   assume kernel_h <= Ckh, kernel_w <= Ckw
//   Bow,Boh = 32,8
//   Cic,Ckh,Ckw = 1,16,16 or 4,8,8
//
template <bool swapkernel, int Bow, int Boh, int Cic, int Ckh, int Ckw>
  __global__ void _conv2generic(float *input, float *kernel, float *output,
				int input_c, int input_h, int input_w,
				int kernel_n, int kernel_h, int kernel_w,
				int stride_h, int stride_w,
				int minibatch) {

  // process Boh*Bow output pixels for every Cic input channels
  __shared__ float sh_kernel[Cic][Ckh][Ckw]; // 1*16*16*4Bytes = 1KB
  // __shared__ float sh_output[Boh][Bow]; // 32*8*4Bytes = 1KB
  
  // output dimensions
  const int output_h = (input_h - kernel_h) / stride_h + 1;
  const int output_w = (input_w - kernel_w) / stride_w + 1;
  const int output_c = kernel_n / input_c; 

  const int oc = blockIdx.x % output_c;
  const int mb = blockIdx.y % minibatch;

  const int ow = threadIdx.x + (blockIdx.x / output_c) * Bow;
  const int oh = threadIdx.y + (blockIdx.y / minibatch) * Boh;

  // input[mb][oh*][ow*]
  // kernel[oc][*][*][*]
  // output[mb][oc][*][*]
  input  += mb*input_c*input_h*input_w + oh*stride_h*input_w + ow*stride_w;
  kernel += oc*input_c*kernel_h*kernel_w;
  output += mb*output_c*output_h*output_w + oc*output_h*output_w;

  // init output in smem
//  sh_output[threadIdx.y][threadIdx.x] = 0; 
//  __syncthreads();

  int ic,cc,kh,kw;
  if ((ow < output_w) && (oh < output_h)) {
    // conv every Cic channels
    float ccsum = 0;
    for (ic = 0; ic < input_c; ic+=Cic) {
      // read kernel[ic->ic+Cic][*][*] into smem
      for (kh = threadIdx.y; kh < kernel_h; kh+=Boh) {
	for (kw = threadIdx.x; kw < kernel_w; kw+=Bow) {
	  for (cc = 0; cc < Cic; ++cc) {
	    sh_kernel[cc][kh][kw] = 
	      kernel[(ic+cc)*kernel_h*kernel_w + kh*kernel_w + kw];
	  }
	}
      }
      __syncthreads();
   
      // conv
      for (kh = 0; kh < kernel_h; ++kh) {
        for (kw = 0; kw < kernel_w; ++kw) {
          float *_input = input + input_w*kh + kw;
          for (cc = 0; cc < Cic; ++cc) {
            ccsum += sh_kernel[cc][kh][kw] * _input[cc*input_h*input_w];
	  }
	}
      }
      // sh_output[threadIdx.y][threadIdx.x] += ccsum;
    } // ic

    output[oh*output_w+ow] += ccsum; // sh_output[threadIdx.y][threadIdx.x]; // cache hit? ..
//    __syncthreads();

  } // if inside
} //////////////////////////////////////////////////////////////////////////////

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

  const int Bow = 32; const int Boh = 8;
  dim3 blocks(nOutputPlane*DIVUP(nOutputRows,Bow),
	      1*DIVUP(nOutputCols,Boh));
  dim3 threads(Bow,Boh);

  _conv2generic <false, Bow, Boh, 4, 8, 8> <<<blocks, threads>>> 
    (input, kernel, output,
     nInputPlane, nInputRows, nInputCols,
     nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
     1, 1, 1); 

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
  int nBatch(128);
  long nInputPlane(32), nInputRows(32), nInputCols(32); // no padding
  long nKernelRows(5), nKernelCols(5);
  long nOutputPlane(128), nOutputRows(28), nOutputCols(28); // stride = 1

//  nKernelRows = nKernelCols = 15;
//  nOutputRows = nOutputCols = nInputRows - nKernelRows + 1;

  float *input = cuda_new(nBatch*nInputPlane*nInputRows*nInputCols);
  float *kernel = cuda_new(nOutputPlane*nInputPlane*nKernelRows*nKernelCols);
  float *output = cuda_new(nBatch*nOutputPlane*nOutputRows*nOutputCols);

  const int Bow = 32; const int Boh = 8;
  dim3 blocks(nOutputPlane*DIVUP(nOutputRows,Bow),
	      nBatch*DIVUP(nOutputCols,Boh));
  dim3 threads(Bow,Boh);  

  _conv2generic <false, Bow, Boh, 4, 16, 16> <<<blocks, threads>>> 
//  _conv2generic <false, Bow, Boh, 4, 8, 8> <<<blocks, threads>>> 
    (input, kernel, output,
     nInputPlane, nInputRows, nInputCols,
     nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
     1, 1, nBatch); 

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
