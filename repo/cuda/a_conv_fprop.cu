
#include <assert.h>
#include <cuda_header.h>

template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache,
          bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_sparse(float* images, float* filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
                                       const int numGroups, 
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv) {
    __shared__ float shFilters[B_Y*colorCache][B_Y * filtersPerThread]; // pre-load B_Y pixels from B_Y*filtersPerThread filters
    __shared__ float shImages[B_Y*colorCache][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int numFilterColors = numImgColors / numGroups;
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

    const int numModules = numModulesX * numModulesY;
    const int blockColorIdx = numFilterColors * blockGroupIdx;

    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

    images += blockColorIdx * imgPixels * imgStride + myImgIdx;
    filters +=blockFilterIdx
            + shFilterLoadY * numFilters + shFilterLoadX;

    if (!conv) {
        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
    }

    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y) * numImages * numModules
            + myImgIdx;

    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }
    
    for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)
        for (int p = 0; p < filterPixels; p += B_Y) {
            // Load B_Y pixels from B_Y*filtersPerThread filters
            if (shFilterLoadY < B_Y) {
                #pragma unroll
                for (int p2 = 0; p2 < B_Y; p2 += B_X/filtersPerThread) {
                    if (p + p2 + shFilterLoadY < filterPixels) {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[((oc+c) * filterPixels + p + p2) * numFilters];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = 0;
                        }
                    }
                }
            }

            // Load B_Y pixels from B_X*imgsPerThread images
            const int pixIdx = p + threadIdx.y;
            if (pixIdx < filterPixels) {
                const int x = imgLoadModPosX + pixIdx % filterSize;
                const int y = imgLoadModPosY + pixIdx / filterSize;
                if (y >= 0 && y < imgSizeY && x >= 0 && x < imgSizeX) {
                    float* m = &images[imgStride * (oc * imgPixels + y * imgSizeX + x)];
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                            #pragma unroll
                            for (int c = 0; c < colorCache; c++) {
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = m[c * imgStride * imgPixels + i * B_X];
                            }
                        } else {
                            #pragma unroll
                            for (int c = 0; c < colorCache; c++) {
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                            }
                        }
                    }
                } else { // Padding
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                        }
                    }
                }
            }

            __syncthreads();

            #pragma unroll
            for (int i = 0; i < B_Y*colorCache; i++) {
                #pragma unroll
                for(int f = 0; f < filtersPerThread; f++) {
                    #pragma unroll
                    for(int g = 0; g < imgsPerThread; g++) {
                        prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y];
                    }
                }

            }

            __syncthreads();
        }
    }

    if (scale) {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModules] + scaleOutputs * prod[f][g];
                }
            }
        }
    } else {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleOutputs * prod[f][g];
                }
            }
        }
    }

}

/* mb = 1
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters)
 * targets:     (numFilters, numModules, numImages)
 */
void _filterActs_test1() {
  long nInputPlane(32), nInputRows(32), nInputCols(32); // no padding
  long nKernelRows(5), nKernelCols(5);
  long nOutputPlane(128), nOutputRows(28), nOutputCols(28); // stride = 1

  float *input = cuda_new(nInputPlane*nInputRows*nInputCols);
  float *kernel = cuda_new(nOutputPlane*nInputPlane*nKernelRows*nKernelCols);
  float *output = cuda_new(nOutputPlane*nOutputRows*nOutputCols);

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
  int numImages = 1; // mb = 1 !
  int imgPixels = nInputRows*nInputCols;
  int imgSizeX = imgPixels / imgSizeY; // X->col
  int filterModuleMult = 1; // conv

  assert(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 2 == 0)));
  assert(numGroups == 1 || numFilterColors % 2 == 0);
  assert(numFilters % (16 * numGroups) == 0);
  assert(numImgColors % numGroups == 0);
  assert(imgSizeY * imgSizeX == imgPixels);
  int numFiltersPerGroup = numFilters / numGroups;

  int imgStride = 1; // stride = 1

  int filterPixels = nKernelRows * nKernelCols;
  int filterSize = int(sqrt(filterPixels));
  assert(filterSize * filterSize == filterPixels); // SQUARE !
  
  int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1; // = 1
  dim3 blocks = numFiltersPerGroup % 32 == 0 ? 
    dim3(DIVUP(numImages, 32 * imgsPerThread), (numModules * numFilters) / (4 * 8)) : 
    dim3(DIVUP(numImages, 32 * imgsPerThread), (numModules * numFilters) / (4 * 4));
  dim3 threads(32, 4);
  bool checkImgBounds = numImages % (32*imgsPerThread) != 0;

  assert(checkImgBounds == true);
  assert(imgsPerThread == 1);
  assert(scaleTargets == 0);
  assert(numFiltersPerGroup % 32 == 0);
  cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 1, 8, 2, false, true >, 
			 cudaFuncCachePreferShared);
  
  filterActs_YxX_sparse < 4, 32, 1, 8, 2, false, true > <<<blocks, threads>>> (input, kernel, output, 
     numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, 
     moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, 
     scaleTargets, scaleOutput, conv);

  cuda_free(input);
  cuda_free(kernel);
  cuda_free(output);

  cudaDeviceSynchronize();  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in _filterActs_test1: %s\n", cudaGetErrorString(err));
  }
}

/* mb = 128
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters)
 * targets:     (numFilters, numModules, numImages)
 */
void _filterActs_test2() {
  long nBatch(128);
  long nInputPlane(32), nInputRows(32), nInputCols(32); // no padding
  long nKernelRows(5), nKernelCols(5);
  long nOutputPlane(128), nOutputRows(28), nOutputCols(28); // stride = 1

  float *input = cuda_new(nBatch*nInputPlane*nInputRows*nInputCols);
  float *kernel = cuda_new(nOutputPlane*nInputPlane*nKernelRows*nKernelCols);
  float *output = cuda_new(nBatch*nOutputPlane*nOutputRows*nOutputCols);

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

  int imgStride = 1; // stride = 1

  int filterPixels = nKernelRows * nKernelCols;
  int filterSize = int(sqrt(filterPixels));
  assert(filterSize * filterSize == filterPixels); // SQUARE !
  
  int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1; // = 1
  dim3 blocks = numFiltersPerGroup % 32 == 0 ? 
    dim3(DIVUP(numImages, 32 * imgsPerThread), (numModules * numFilters) / (4 * 8)) : 
    dim3(DIVUP(numImages, 32 * imgsPerThread), (numModules * numFilters) / (4 * 4));
  dim3 threads(32, 4);
  bool checkImgBounds = numImages % (32*imgsPerThread) != 0;

  assert(checkImgBounds == false);
  assert(imgsPerThread == 4);
  assert(scaleTargets == 0);
  assert(numFiltersPerGroup % 32 == 0);

  cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 8, 2, false, false >, cudaFuncCachePreferShared);
  filterActs_YxX_sparse < 4, 32, 4, 8, 2, false, false > <<<blocks, threads>>>
    (input, kernel, output,
     numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, 
     moduleStride, numModulesY, numModulesX, imgStride, numImgColors, 
     numGroups, scaleTargets, scaleOutput, conv);

  cuda_free(input);
  cuda_free(kernel);
  cuda_free(output);

  cudaDeviceSynchronize();  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in _filterActs_test1: %s\n", cudaGetErrorString(err));
  }
}

int main() {
  cuda_init();
  clock_t t;
  for (int i = 0; i < 10; ++i) {
    tic(&t,"_filterActs_test1...");
    _filterActs_test1();
    toc(&t, "done");

    tic(&t,"_filterActs_test2...");
    _filterActs_test2();
    toc(&t, "done");
  }   
  return 0;
}
