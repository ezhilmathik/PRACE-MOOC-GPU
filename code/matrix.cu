//-*-C++-*-
#include<iostream>

using namespace std;

__global__ void matrix_mul(float* d_a, float* d_b,
			   float* d_c, int Width) 
{
  
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
  if ((tx < Width) && (ty < Width)) 
    {
      float Pvalue = 0;
      // each thread computes one 
      // element of the block sub-matrix
      for (int k = 0; k < Width; ++k) 
	{
	  Pvalue += d_a[tx*Width+k]*d_b[k*Width+ty];
	}
      d_c[tx*Width+ty] = Pvalue;
    }
}

int main()
{
  
  cout << "Programme assumes that matrix size is N*N "<<endl;
  cout << "Please enter the N size number "<< endl;
  int N=0;
  cin >> N;

  // Initialize the memory on the host
  float *a, *b, *c;       
  
  // Initialize the memory on the device
  float *d_a, *d_b, *d_c; 
  
  // Allocate host memory
  a   = (float*)malloc(sizeof(float) * (N*N));
  b   = (float*)malloc(sizeof(float) * (N*N));
  c   = (float*)malloc(sizeof(float) * (N*N));
  
  // Initialize host arrays
  for(int i = 0; i < (N*N); i++)
    {
      a[i] = 2.0f;
      b[i] = 2.0f;
    }
  
  // Allocate device memory
  cudaMalloc((void**)&d_a, sizeof(float) * (N*N));
  cudaMalloc((void**)&d_b, sizeof(float) * (N*N));
  cudaMalloc((void**)&d_c, sizeof(float) * (N*N));
  
  // Transfer data from host to device memory
  cudaMemcpy(d_a, a, sizeof(float) * (N*N), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(float) * (N*N), cudaMemcpyHostToDevice);
  
  // Thread organization
  int blockSize = 32;
  dim3 dimBlock(blockSize,blockSize,1);
  dim3 dimGrid(ceil(N/float(blockSize)),ceil(N/float(blockSize)),1);
  
  // Device fuction call 
  matrix_mul<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, N);

  // Transfer data back to host memory
  cudaMemcpy(c, d_c, sizeof(float) * (N*N), cudaMemcpyDeviceToHost);

  // Verification
  for(int i = 0; i < N; i++)
    {
      for(int j = 0; j < N; j++)
      	{
	  cout << c[j] <<" ";
	}
      cout << " " <<endl;
    }
  
  // Deallocate device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  
  // Deallocate host memory
  free(a); 
  free(b); 
  free(c);

  return 0;
}
