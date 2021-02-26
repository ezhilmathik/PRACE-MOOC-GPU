//-*-C++-*-
#include<iostream>
#define TILE_WIDTH 3

using namespace std;

__device__ int a_block[TILE_WIDTH][TILE_WIDTH];
__device__ int b_block[TILE_WIDTH][TILE_WIDTH];

__global__ void matrix_mul(const float *d_a, const float *d_b, 
			   float *d_c, int width)
{  
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  float single_entry = 0;

  for(int i = 0; i < width / TILE_WIDTH; ++i)
    {
      a_block[ty][tx] = d_a[row * width + (i * TILE_WIDTH + tx)];
      b_block[ty][tx] = d_b[col + (i * TILE_WIDTH + ty) * width];
	 
      __syncthreads(); 
	  
      for(int j = 0; j < TILE_WIDTH; ++j)
	{
	  single_entry += a_block[ty][j] * b_block[j][tx];    
	}
      __syncthreads();
    }
  d_c[row*width+col] = single_entry;
}

int main()
{  
  cout << "Programme assumes that matrix size is N*N "<<endl;
  cout << "Please enter the N size number "<< endl;
  int N;
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

  cudaDeviceSynchronize();
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
