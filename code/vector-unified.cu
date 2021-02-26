//-*-c++-*-
#include <iostream>
#include <math.h>

using namespace std;

// CUDA kernel to add elements of two arrays
__global__
void vector_add(float *x, float *y, int N)
{

  int blockId = blockIdx.x + blockIdx.y * gridDim.x;     
  int i = blockId * (blockDim.x * blockDim.y) +          
    (threadIdx.y * blockDim.x) + threadIdx.x;
  if(i < N)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 20;
  float *x, *y;
  
  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  
  // Initialize x and y arrays on the host
  for (int i = 0; i < N; i++) 
    {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }

  // Prefetch the data to the GPU
  int device = -1;
  cudaGetDevice(&device);
  cudaMemPrefetchAsync(x, N*sizeof(float), device, NULL);
  cudaMemPrefetchAsync(y, N*sizeof(float), device, NULL);
  

  // Thread organization 
  int blockSize = 16;
  dim3 dimBlock(blockSize,blockSize,1);
  dim3 dimGrid(ceil(N/float(blockSize)),ceil(N/float(blockSize)),1);
  
  vector_add<<<dimGrid, dimBlock>>>(x,y,N);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  

  // Verification
  for(int i = 0; i < N/2; i++)
    {
      for(int j = 0; j < N/2; j++)
	{
	  cout << y[j] <<" ";
	}
      cout << " " <<endl;
    }
  
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  cout<< "Max error: " << maxError << endl;
  
  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
