//-*-C++-*-
#include<iostream>

//#define TILE_WIDTH 10
//#define TILE_DIM 100

using namespace std;

__global__
void matrix_mul(float *a, float* b, float *c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    //    if (row < TILE_DIM && col < TILE_DIM)
    if (row < N && col < N)
      {
	//	for (int i = 0; i < TILE_DIM; i++)
	for (int i = 0; i < N; i++)
	  {
	    //	    sum += a[row*TILE_DIM+i] * b[i*N+col];
	    sum += a[row*N+i] * b[i*N+col];
	  }
	c[row*N+col] = sum;
      }
}


/*
//__device__ float aTile[TILE_WIDTH][TILE_WIDTH];
__global__ void matrix_mul(float *a, float* b, float *c,
int width)
{

  __shared__ float aTile[TILE_DIM][TILE_DIM];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  //int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  //int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  
  float single_entry = 0.0f;
  aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
  __syncwarp();
  if (row < width && col < width)
    {
      for (int i = 0; i < width; i++) 
	{
	  single_entry += aTile[threadIdx.y][i]* b[i*width+col];
	  //single_entry += a[row*width+col] *  b[i*width+col];
	}
      c[row*width+col] = single_entry;

    }
}
*/


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
      c[i] = 2.0f;
    }
  
  // Allocate device memory
  cudaMalloc((void**)&d_a, sizeof(float) * (N*N));
  cudaMalloc((void**)&d_b, sizeof(float) * (N*N));
  cudaMalloc((void**)&d_c, sizeof(float) * (N*N));
  
  // Transfer data from host to device memory
  cudaMemcpy(d_a, a, sizeof(float) * (N*N), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(float) * (N*N), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, sizeof(float) * (N*N), cudaMemcpyHostToDevice);
  
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
	  cout << c[j*N+i] <<" ";
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
