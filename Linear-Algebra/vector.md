### Basic Vector Operations

Computers are mainly invented and grown a tremendous achievement to do numerical computation, in particular, vector and matrix operations. We will not look into the basic vector operations. 

~~~bash
//-*-C++-*-
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#define N 5120
#define MAX_ERR 1e-6

// CPU function that adds two vector 
float * vector_add(float *out, float *a, float *b, int n) 
{
  for(int i = 0; i < n; i ++)
    {
      out[i] = a[i] + b[i];
    }
  return out;
}

int main()
{
  // Initialize the memory on the host
  float *a, *b, *out;       
  
  // Allocate host memory
  a   = (float*)malloc(sizeof(float) * N);
  b   = (float*)malloc(sizeof(float) * N);
  out = (float*)malloc(sizeof(float) * N);
  
  // Initialize host arrays
  for(int i = 0; i < N; i++)
    {
      a[i] = 1.0f;
      b[i] = 2.0f;
    }
    
  // Start measuring time
  clock_t start = clock();

  // Executing CPU funtion 
  vector_add(out, a, b, N);

  // Stop measuring time and calculate the elapsed time
  clock_t end = clock();
  double elapsed = double(end - start)/CLOCKS_PER_SEC;
        
  printf("Time measured: %.3f seconds.\n", elapsed);
  
  // Verification
  for(int i = 0; i < N; i++)
    {
      assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

  printf("out[0] = %f\n", out[0]);
  printf("PASSED\n");
    
  // Deallocate host memory
  free(a); 
  free(b); 
  free(out);

  return 0;
}
~~~


~~~bash
//-*-C++-*-
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 5120
#define MAX_ERR 1e-6


#define DEBUG
#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)
////////////////////////////////////////////////////////////////////////////////
// A method for checking error in CUDA calls
////////////////////////////////////////////////////////////////////////////////
inline void __checkCuda(cudaError_t error, const char *file, const int line)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (error != cudaSuccess)
    {
      printf("checkCuda error at %s:%i: %s\n", file, line,
	     cudaGetErrorString(cudaGetLastError()));
      exit(-1);
    }
#endif
  return;
}

__global__ void vector_add(float *out, float *a, float *b, int n) 
{
 // converting 2D thread structure into 1D thread structure 
  int i = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  // Allow the threads only within the size of N
  if(i < n)
    {
      //printf("Hello thread %d\n", k);
      out[i] = a[i] + b[i];
    }

  // Synchronice all the threads 
  __syncthreads();
}

int main()
{

  // Initialize the memory on the host
  float *a, *b, *out;       

  // Initialize the memory on the device
  float *d_a, *d_b, *d_out; 
  
  // Allocate host memory
  a   = (float*)malloc(sizeof(float) * N);
  b   = (float*)malloc(sizeof(float) * N);
  out = (float*)malloc(sizeof(float) * N);
  
  // Initialize host arrays
  for(int i = 0; i < N; i++)
    {
      a[i] = 1.0f;
      b[i] = 2.0f;
    }
  
  // Allocate device memory
  cudaMalloc((void**)&d_a, sizeof(float) * N);
  cudaMalloc((void**)&d_b, sizeof(float) * N);
  cudaMalloc((void**)&d_out, sizeof(float) * N);
  
  // Transfer data from host to device memory
  cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

  // Thread organization 
  dim3 comp_G(80, 80, 1);    
  dim3 comp_B(8, 8, 1); 

  cudaEvent_t start, stop;

  // cuda event initialize 
  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));

  // cuda event start
  checkCuda(cudaEventRecord(start , 0));
    
  // Executing kernel 
  vector_add<<<comp_G, comp_B>>>(d_out, d_a, d_b, N);
  
  // stop measuring the event‏
  checkCuda(cudaEventRecord(stop , 0));

  // close the initialization
  checkCuda(cudaEventSynchronize( start ));
  checkCuda(cudaEventSynchronize( stop ));
  
  // total time consumption
  float dt_ms;
  checkCuda(cudaEventElapsedTime(&dt_ms, start, stop));
  printf(" Time taken for the GPU is ---------------- %f seconds\n", dt_ms/1000);

  // Transfer data back to host memory
  cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
  
  // Verification
  for(int i = 0; i < N; i++)
    {
      assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

  printf("out[0] = %f\n", out[0]);
  printf("PASSED\n");
  
  // Deallocate device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  
  // Deallocate host memory
  free(a); 
  free(b); 
  free(out);
}

~~~
