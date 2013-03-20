
#include "time.h"
#include "math.h"
#include "stdio.h"

#include "cuda.h"
#include "cublas.h"
#include "cuda_runtime.h"

#define gFloat float

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

gFloat * cuda_new(int sz) {
  gFloat *loc;
  cublasStatus stat = cublasAlloc(sz, sizeof(gFloat), (void**)&loc);
  // rand init
//  gFloat cpu[sz];
//  for (int i = 0; i < sz; ++i) cpu[i] = rand();
//  stat = cublasSetVector(sz, sizeof(gFloat), cpu, 1, loc, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("error: cuda new failed");
    return NULL;
  }
  return loc;
}

void cuda_free(float *loc) {
  if (loc != NULL)
    cublasFree(loc);
}

void cuda_init(int gpuid=0) {
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  if ((gpuid < 0) || (gpuid >= num_devices)) { // max id: num_devices-1;
    gpuid = 0; 
  }
  printf("use gpuid: %d\n", gpuid);
  cudaSetDevice(gpuid);
  cublasInit();
}

void tic(clock_t *t, const char *msg) {
    *t = clock();
    printf("%s...", msg);
}

void toc(clock_t *t, const char *msg="") {
    *t = clock() - *t;
    printf("%s in %f secs.\n", msg, ((float)(*t))/CLOCKS_PER_SEC);
}

