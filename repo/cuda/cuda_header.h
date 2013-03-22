
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

gFloat * cuda_new(int sz, bool rnd = false, float value = 0) {
  gFloat *loc;
  cublasStatus stat = cublasAlloc(sz, sizeof(gFloat), (void**)&loc);
  gFloat *cpu = new float[sz];
  // rand init
  if (rnd) {
    for (int i = 0; i < sz; ++i) cpu[i] = (float) rand() / RAND_MAX;
  }
  else {
    for (int i = 0; i < sz; ++i) cpu[i] = value;
  }
  stat = cublasSetVector(sz, sizeof(gFloat), cpu, 1, loc, 1);
  delete[] cpu;
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

void cuda_print(float *gpu, int offset, int sz) {
    float *cpu = new float[sz];
    gpu += offset;
    cublasGetVector(sz, sizeof(float), gpu, 1, cpu, 1);
    for (int i = 0; i < sz; ++i)
      printf("%d:%f\n", i+offset, cpu[i]);
    delete[] cpu;
}

void tic(clock_t *t, const char *msg) {
    *t = clock();
    printf("%s...", msg);
}

void toc(clock_t *t, const char *msg="") {
    *t = clock() - *t;
    printf("%s in %f secs.\n", msg, ((float)(*t))/CLOCKS_PER_SEC);
}

