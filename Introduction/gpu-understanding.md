### Understanding Your GPUs

Most of the time, when we run our GPU (CUDA) code, we do not know exactly what is the Nvidia GPU type we run. Understanding your GPU card will give know information that will be useful for writing optimized and efficient code. 

~~~bash
//-*-C++-*-
#include <stdio.h>
 
// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
  printf("Major revision number:         %d\n",  devProp.major);
  printf("Minor revision number:         %d\n",  devProp.minor);
  printf("Name:                          %s\n",  devProp.name);
  printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
  printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
  printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
  printf("Warp size:                     %d\n",  devProp.warpSize);
  printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
  printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
  for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
  for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
  printf("Clock rate:                    %d\n",  devProp.clockRate);
  printf("Total constant memory:         %u\n",  devProp.totalConstMem);
  printf("Texture alignment:             %u\n",  devProp.textureAlignment);
  printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
  printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
  printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
  return;
}
 
int main()
{
  // Number of CUDA devices
  int devCount;
  cudaGetDeviceCount(&devCount);
  printf("CUDA Device Query...\n");
  printf("There are %d CUDA devices.\n", devCount);
 
  // Iterate through devices
  for (int i = 0; i < devCount; ++i)
    {
      // Get device properties
      printf("\nCUDA Device #%d\n", i);
      cudaDeviceProp devProp;
      cudaGetDeviceProperties(&devProp, i);
      printDevProp(devProp);
    }
 
  printf("\nPress any key to exit...");
  char c;
  scanf("%c", &c);
 
  return 0;
}
~~~

How to compile:

~~~bash
nvcc -arch=compute_70 device-quary.cu
~~~bash

~~~bash 
CUDA Device Query...
There are 1 CUDA devices.

CUDA Device #0
Major revision number:         7
Minor revision number:         0
Name:                          Tesla V100-SXM2-16GB
Total global memory:           4060610560
Total shared memory per block: 49152
Total registers per block:     65536
Warp size:                     32
Maximum memory pitch:          2147483647
Maximum threads per block:     1024
Maximum dimension 0 of block:  1024
Maximum dimension 1 of block:  1024
Maximum dimension 2 of block:  64
Maximum dimension 0 of grid:   2147483647
Maximum dimension 1 of grid:   65535
Maximum dimension 2 of grid:   65535
Clock rate:                    1530000
Total constant memory:         65536
Texture alignment:             512
Concurrent copy and execution: Yes
Number of multiprocessors:     80
Kernel execution timeout:      No
~~~
