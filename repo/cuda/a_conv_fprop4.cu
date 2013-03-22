
#include <assert.h>
#include <cuda_header.h>
#include <cuda_convker.h>
#include <cublas_v2.h>

// assume: shKerLoads >= 1, shImgLoads >= 1, assert(mod==0)
template < int B_Y, int B_X, int imgsPerThread, int filtersPerThread,
           int colorCache, int pixelCache, bool checkImgBounds >
  __global__ void _filterActs_YxX_sparse
  (
   float* images, float* filters, float* targets,
   const int numImages, const int numFilters,
   const int imgSizeY, const int imgSizeX, 
   const int filterSize, const int paddingStart,
   const int moduleStride, const int numModulesY, const int numModulesX, 
   const int imgStride, const int numImgColors,
   const int numGroups, const float scaleTargets, const float scaleOutputs,
   const bool conv) {

  const int imgPixels = imgSizeY * imgSizeX;
  const int filterPixels = filterSize * filterSize;
  const int numModules = numModulesX * numModulesY;
  const int numThreads = B_X * B_Y;

  const int filtersPerBlock = filtersPerThread * B_Y;
  const int imagesPerBlock = imgsPerThread * B_X;
  const int blocksPerModule = DIVUP(numFilters, filtersPerBlock); // ceil !
  const int shPixelsColors = pixelCache * colorCache;

  __shared__ float shFilters[shPixelsColors][filtersPerBlock];
  __shared__ float shImages[shPixelsColors][imagesPerBlock]; 

  const int moduleIdx = blockIdx.y / blocksPerModule;
  const int blockFilterIdx = filtersPerBlock * (blockIdx.y % blocksPerModule);
  const int blockImgIdx = blockIdx.x * B_X * imgsPerThread;

  const int tidx = threadIdx.y * B_X + threadIdx.x;
  
  const int imgLoadModPosY = 
    paddingStart + (moduleIdx / numModulesX) * moduleStride;
  const int imgLoadModPosX = 
    paddingStart + (moduleIdx % numModulesX) * moduleStride;

  const int shLoadY = tidx / pixelCache; // which depth, which image
  const int shLoadX = tidx % pixelCache; // which pixel
  const int shLoads = numThreads/pixelCache; // how many per 'cycle'

  images += blockImgIdx * numImgColors * imgPixels; // + shLoadX;  
  filters += blockFilterIdx * numImgColors * filterPixels; //  + shLoadX;

  targets += moduleIdx + 
  	     (blockFilterIdx + threadIdx.y) * numModules +
    	     (blockImgIdx + threadIdx.x) * numModules * numFilters;
  
  // init
  float prod[filtersPerThread][imgsPerThread];
#pragma unroll
  for(int f = 0; f < filtersPerThread; f++) {
#pragma unroll
    for(int g = 0; g < imgsPerThread; g++) {
      prod[f][g] = 0;
    }
  }

  const int shFilters_sz = filtersPerBlock*shPixelsColors;
  const int shFilters_inc = shFilters_sz/numThreads;
  for (int sker = tidx; sker < shFilters_sz; sker += shFilters_inc) 
    shFilters[sker%shPixelsColors][sker/shPixelsColors] = 0.0;

  const int shImages_sz = imagesPerBlock*shPixelsColors;
  const int shImages_inc = shImages_sz/numThreads;
  for (int simg = tidx; simg < shImages_sz; simg += shImages_inc) 
    shImages[simg%shPixelsColors][simg/shPixelsColors] = 0.0;
  
  __syncthreads();

  // CONVOLUTION
  for (int cc = 0; cc < numImgColors; cc += colorCache) {
    for (int cp = 0; cp < filterPixels; cp += pixelCache) { 
      // load filters in bound
      if ((cp + shLoadX) < filterPixels) {
        // TODO generalize to non-square
        float* ker = &filters[cc*filterPixels+cp+shLoadX];
        for (int d = shLoadY; d < filtersPerBlock; d += shLoads) {
  	  #pragma unroll
	  for (int c = 0; c < colorCache; ++c) {
            shFilters[c*B_Y+shLoadX][d] = 
	    	ker[(d*numImgColors+c)*filterPixels];
	  }
        }
      } else {
        for (int d = shLoadY; d < filtersPerBlock; d += shLoads) {
          #pragma unroll       
          for (int c = 0; c < colorCache; ++c) {
            shFilters[c*B_Y+shLoadX][d] = 0.0;
          }
        }
      }

      // load images in bound
      const int x = imgLoadModPosX + (cp + shLoadX) % filterSize; 
      const int y = imgLoadModPosY + (cp + shLoadX) / filterSize;
      if (y >= 0 && y < imgSizeY && x >= 0 && x < imgSizeX) {
        float* img = &images[cc*imgPixels+y*imgSizeX+x];
        for (int z = shLoadY; z < imagesPerBlock; z += shLoads) {
	  #pragma unroll
	  for (int c = 0; c < colorCache; c++) {
	    shImages[c*B_Y+shLoadX][z] = img[(z*numImgColors+c)*imgPixels];
	  }
	}
      } else {
/*
        for (int z = shLoadY; z < imagesPerBlock; z += shImgLoads) {
          #pragma unroll
          for (int c = 0; c < colorCache; c++) {
            shImages[z][c*B_Y+shLoadX] = 0.0;
          }
        }
*/
      }

      __syncthreads();

      // conv
      #pragma unroll
      for (int i = 0; i < pixelCache*colorCache; ++i) {
	#pragma unroll
	for (int f = 0; f < filtersPerThread; ++f) {
	  #pragma unroll
	  for (int g = 0; g < imgsPerThread; ++g) {
	    // CLOCK HERE, NON SEQ ACCESS
	    prod[f][g] += 
	      shImages[i][g*B_X+threadIdx.x]*shFilters[i][f*B_Y+threadIdx.y];
	  }
	}
      }

      __syncthreads();
    }
  }

  //#pragma unroll
  for (int g = 0; g < imgsPerThread; g++) {
    // checkImgBounds
    if (blockImgIdx + threadIdx.x + g*B_X < numImages) { 
      //#pragma unroll
      for (int f = 0; f < filtersPerThread; f++) {
        // checkTgtBounds
	if (blockFilterIdx + threadIdx.y + f*B_Y < numFilters)
	  /// SLOW WRITE OUT
          targets[g*B_X*numModules*numFilters + f*B_Y*numModules] = prod[f][g];
      }
    }
  }
}

void _filterActs_test128(long nBatch, 
                         long nInputPlane, long nInputRows, long nInputCols, 
                         long nKernelRows, long nKernelCols, 
                         long nOutputPlane, long nOutputRows, long nOutputCols) {
  assert(nOutputRows == nInputRows - nKernelRows + 1); // TODO test STRIDE
  assert(nOutputCols == nInputCols - nKernelCols + 1);

  float *input = 
    cuda_new(nBatch*nInputPlane*nInputRows*nInputCols, true, 1);
  float *kernel = 
    cuda_new(nOutputPlane*nInputPlane*nKernelRows*nKernelCols, true, 1);
  float *output = 
    cuda_new(nBatch*nOutputPlane*nOutputRows*nOutputCols, false, -1);

  int imgSizeY = nInputRows; // Y->row
  int paddingStart = 0;
  int moduleStride = 1;
  int numImgColors = nInputPlane;
  int numGroups = 1;
  float scaleTargets = 0;
  float scaleOutput = 1;
  bool conv = true;
  int numModulesY = nOutputRows;
  int numModulesX = nOutputCols;

  int numFilterColors = numImgColors / numGroups;
  int numFilters = nOutputPlane;
  int numModules = numModulesY * numModulesX;
  int numImages = nBatch; // mb = 128 !
  int imgPixels = nInputRows*nInputCols;
  int imgSizeX = imgPixels / imgSizeY; // X->col
  int filterModuleMult = 1; // conv

  assert(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 2 == 0)));
  assert(numGroups == 1 || numFilterColors % 2 == 0);
  assert(numFilters % (16 * numGroups) == 0);
  assert(numImgColors % numGroups == 0);
  assert(imgSizeY * imgSizeX == imgPixels);
  int numFiltersPerGroup = numFilters / numGroups;

  int imgStride = nBatch;

  int filterPixels = nKernelRows * nKernelCols;
  int filterSize = int(sqrt(filterPixels));
  assert(filterSize * filterSize == filterPixels); // SQUARE !

  const int B_X = 32;
  const int B_Y = 4; 
  const int imgsPerThread = 4; // 4
  const int filtersPerThread = 8; //8
  const int colorCache = 1;
  const int pixelCache = 8; // 8

  dim3 blocks = dim3(DIVUP(numImages, B_X*imgsPerThread), 
       	      	     numModules * DIVUP(numFilters, B_Y * filtersPerThread));
  dim3 threads(B_X,B_Y);

  bool checkImgBounds = true; // numImages % (32*imgsPerThread) != 0;
  assert(imgsPerThread == 4);
  assert(scaleTargets == 0);
  assert(numFiltersPerGroup % 32 == 0);

  clock_t t;
  cudaFuncSetCacheConfig(_filterActs_YxX_sparse< B_Y, B_X, imgsPerThread, filtersPerThread, colorCache, pixelCache, true >, cudaFuncCachePreferShared);
  cudaDeviceSynchronize();
  tic(&t,"_filterActs_YxX_sparse...");
  _filterActs_YxX_sparse < B_Y, B_X, imgsPerThread, filtersPerThread, colorCache, pixelCache, true > <<<blocks, threads>>>
    (input, kernel, output,
     numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, 
     moduleStride, numModulesY, numModulesX, imgStride, numImgColors, 
     numGroups, scaleTargets, scaleOutput, conv);
  cudaDeviceSynchronize();
  toc(&t, "done");

////////////////////////////TEST CORRECTNESS///////////////////////////////////
  float *input0 = cuda_new(nBatch*nInputPlane*nInputRows*nInputCols);
  float *kernel0 = cuda_new(nOutputPlane*nInputPlane*nKernelRows*nKernelCols);
  float *output0 = cuda_new(nBatch*nOutputPlane*nOutputRows*nOutputCols);

  cublasHandle_t handle;
  cublasCreate(&handle);
  const float _a = 1, _b = 0;
  int _r, _c;

  // input0 = input'
  _r = nInputPlane*nInputRows*nInputCols;
  _c = nBatch;
  cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, _c, _r,
              &_a, input, _r, &_b, input, _r, input0, _c); 

  // kernel0 = kernel'
  _r = nInputPlane*nKernelRows*nKernelCols;
  _c = nOutputPlane;
  cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, _c, _r,    
              &_a, kernel, _r, &_b, kernel, _r, kernel0, _c);
  cudaDeviceSynchronize();

  blocks = numFiltersPerGroup % 32 == 0 ? 
    dim3(DIVUP(numImages, 32 * imgsPerThread), (numModules * numFilters) / (4 * 8)) : 
    dim3(DIVUP(numImages, 32 * imgsPerThread), (numModules * numFilters) / (4 * 4));
  threads = dim3(32,4);
  cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 8, 2, false, true >, cudaFuncCachePreferShared);
  cudaDeviceSynchronize();
  tic(&t,"filterActs_YxX_sparse...");
  filterActs_YxX_sparse < 4, 32, 4, 8, 2, false, true > <<<blocks, threads>>>
   (input0, kernel0, output0,
    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart,
    moduleStride, numModulesY, numModulesX, imgStride, numImgColors,
    numGroups, scaleTargets, scaleOutput, conv);
  cudaDeviceSynchronize();
  toc(&t, "done");

  //  printf("\nalex output\n");
  //  cuda_print(output0, 0, 32); // *nOutputRows*nOutputCols);

  // _o = output'
  float *_o = cuda_new(nBatch*nOutputPlane*nOutputRows*nOutputCols);
  _r = nOutputPlane*nOutputRows*nOutputCols;
  _c = nBatch;
  cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, _c, _r,
              &_a, output, _r, &_b, output, _r, _o, _c); 
  cudaDeviceSynchronize();

  //  printf("\nmy output\n");
  //  cuda_print(output, 0, 32); // 128*128*28*28);

  //  printf("\nmy output to alex\n");
  //  cuda_print(_o, 0, 32);
  
  const float _d = -1;
  cublasSaxpy(handle, _r*_c, &_d, output0, 1, _o, 1); // _o -= output0
  float _residue(-1);
  cublasSnrm2(handle, _r*_c, _o, 1, &_residue);
  printf("residue: %f\n", _residue);

  cuda_free(_o);

  cuda_free(input0);
  cuda_free(kernel0);
  cuda_free(output0);
///////////////////////////END OF TEST CORRECTNESS/////////////////////////////

  cuda_free(input);
  cuda_free(kernel);
  cuda_free(output);

  cudaDeviceSynchronize();  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in _filterActs_test128: %s\n", cudaGetErrorString(err));
  }
}

void _filterActs_case0() {   
  _filterActs_test128(128,32,32,32,5,5,128,28,28);
  _filterActs_test128(32,2,128,128,29,29,32,100,100);
  _filterActs_test128(32,2,8,8,2,2,32,7,7);
}

int main() {
  cuda_init();
  for (int i = 0; i < 3; ++i) {
    _filterActs_case0();
  }
  return 0;
}

