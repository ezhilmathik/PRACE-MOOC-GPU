# Week 2: CUDA (basic): Introduction to CUDA Programming

This week, we will study hello world programming in CUDA programming language and how the threads are organized on the Nvidia GPUs. And continuing with knowing different CUDA API for the Nvidia CUDA programming. Finally, we go through the vector and matrix operations on the Nvidia GPUs using the CUDA programming.


## Day 1

## Basic Programming (hello world)

In this article, we see our first hello world programming using the GPU. We will then go through how the device, kernel, thread synchronize, and device synchronize are organized in the CUDA code.

### CUDA Programming 

In this article, we will study the simple CUDA programming for printing out the hello world. 

### _Hello World_ from GPU

In this below section, we will show how to print the _Hello Wolrd_  program from CUDA Programming. The below code shows the simple _C/C++ programming_ and simple _CUDA programming_ to show the simple difference between _CPU_ and _GPU_ programming. 

####  C/C++:
~~~ cpp
// hello-world.c
#include <stdio.h>                                    
                                                      
void c_function()                                       
{                                                     
  printf("Hello World!\n");                           
}                                                     
                                                      
int main()                                            
{                                                     
  c_function();                                         
  return 0;                                           
}
~~~

#### CUDA:
~~~ cpp
// hello-world.cu 
#include <stdio.h> 
__global__ void cuda_function()
{
   printf("Hello World from GPU!\n"); 
    __syncthreads();               // to synchronize all threads
}

int main()
{
    cuda_function<<<1,1>>>();
    cudaDeviceSynchronize();      // to synchronize device call
    return 0;
}
~~~  

#### Compilation instruction:

~~~bash
# Load the CUDA module (compiler)         # Load the module
$ nvcc -arch=compute_70 hello-world.cu    # Compilation
$ ./a.out                                 # Code execution
~~~

* Here `-arch=compute_70` is a compute architecture for the Nvidia Volta GPUs. Each Nvidia GPUs falls into certain [compute capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities). And this can be defined by using `-arch=compute_XX`

#### The following steps will demonstrate the necessary understanding of the above discussed code:
* The main idea of using `GPU` is that we want to run our sub-task functions in the `GPU`. To do that the function should be declared as `__global__`, this means that the declared function will always run on the `GPUs`.
* And at the same time, it should be also called with threads `<<<-,-,->>>`, from the main program where it is being called. For example, from the above example it should be defined as `cuda_function<<<1,1>>>()` from just as a `c_function()`. 
* Above all, we also need to synchronize the calls (both threads and device calls). Otherwise, we will get the wrong solution in the computation. Please check by yourself by removing the call `cudaDeviceSynchronize()`, you will not be able to print out the `Hello World`. It is mainly due to the `master thread` not waiting for the `slave thread`, in this case, you produce the wrong result in the computation.                    


## Day 2

## Understanding the CUDA Threads

This article will give you a detailed overview of how the CUDA threads are organized and how it is mapped on the Nvidia GPUs. Plus also it shows how to convert different thread blocks of threads into another form. 

## Understanding the CUDA Threads

### We will now look into how these CUDA threads are working. 

* It is really important to understand how the CUDA threads are created and working within the CUDA programming. 

* Within the CUDA programming, threads are organized as 1D, 2D and 3D. We will explain later why we have this methodology in CUDA programming. 

![alt text](https://drive.google.com/uc?export=view&id=1BcHHpYJ3C8iekOOxbmeMkGLqunrDAzTC)





![alt text](https://drive.google.com/uc?export=view&id=1nREMtBBOny7F3phc6iX-FIx2laI7Gh2b)


* The [CUDA occupancy calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html) will provide you with a simple overview of how CUDA threads should be organized. They are pre-computed [here](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/CUDA_Occupancy_Calculator.xls) for different CUDA architecture. Please note, the preconfigured recommendation, does not work well for all the cases. This is just an overview of how the threads should be organized and simple and standard cases. 



| `Memory`  | `Location` | `Cached` | `Device Access` | `Scope`                | `Life Time`    |
|----------|----------|--------|---------------|----------------------|--------------|
| **Register** | On-chip  | N/A    | R/W           | one thread           | thread       |
| **Local**    | DRAM     | Yes**  | R/W           | one thread           | thread       |
| **Shared**   | On-chip  | N/A    | R/W           | all threads in block | thread block |
| **Global**   | DRAM     | *      | R/W           | all threads in host  | Application  |
| **Constant** | DRAM     | Yes    | R             | all threads in host  | Application  |
| **Texture**  | DRAM     | Yes    | R             | all threads in host  | Application  |

    * cached L2 by default by latest compute capabilities
    ** cached L2 by default only on compute capabilities 5.x

#### Grids, Blocks and Threads organization in CUDA:

* In CUDA, threads are organized as Grids, Blocks, and Threads. And they can be defined as 1D, 2D and 3D blocks. For example, a CUDA kernel function can be called as `VecMul<<<1, N>>>(A, B, C);`
* CUDA also provided the builtin function to define the threads.
    * gridDim.x, gridDim.y and gridDim.z are defining the number of the blocks in the grids in x,y and z dimensions.
    * blockIdx.x, blockIdx.y and blockIdx.z are defining the index of the current block within the grid in x, y and z dimensions. 
    * blockDim.x, blockDim.y and blockDim.z are defining the number of threads in the block in x, y and z dimensions. 
    * threadIdx.x, threadIdx.y and threadIdx.z are defining the index of the thread within a block in x, y and z dimension. 

~~~
// Kernel invocation with one block of N * N * 1 threads
int numBlocks = 1;
dim3 threadsPerBlock(N, N);
MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
~~~

* In some situations, the whole thread should be converted to just a single array of thread. But threads can be created as 1D, 2D and 3D. However, at the same time, in a particular application, we just need a single array in the global function. The below list shows an example of different CUDA dimensions into different dimensions. 

~~~bash
//1D grid of 1D blocks
__device__ int getGlobalIdx_1D_1D()
{
  return blockIdx.x * blockDim.x + threadIdx.x;
}

//1D grid of 2D blocks
__device__ int getGlobalIdx_1D_2D()
{
  return blockIdx.x * blockDim.x * blockDim.y
    + threadIdx.y * blockDim.x + threadIdx.x;
}

//1D grid of 3D blocks
__device__ int getGlobalIdx_1D_3D()
{
  return blockIdx.x * blockDim.x * blockDim.y * blockDim.z 
    + threadIdx.z * blockDim.y * blockDim.x
    + threadIdx.y * blockDim.x + threadIdx.x;
}

//2D grid of 1D blocks 
__device__ int getGlobalIdx_2D_1D()
{
  int blockId   = blockIdx.y * gridDim.x + blockIdx.x;
  int threadId = blockId * blockDim.x + threadIdx.x; 
  return threadId;
}

//2D grid of 2D blocks  
 __device__ int getGlobalIdx_2D_2D()
{
  int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
  int threadId = blockId * (blockDim.x * blockDim.y) +
    (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}

//2D grid of 3D blocks
__device__ int getGlobalIdx_2D_3D()
{
  int blockId = blockIdx.x 
    + blockIdx.y * gridDim.x; 
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
    + (threadIdx.z * (blockDim.x * blockDim.y))
    + (threadIdx.y * blockDim.x)
    + threadIdx.x;
  return threadId;
}

//3D grid of 1D blocks
__device__ int getGlobalIdx_3D_1D()
{
  int blockId = blockIdx.x 
    + blockIdx.y * gridDim.x 
    + gridDim.x * gridDim.y * blockIdx.z; 
  int threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}

//3D grid of 2D blocks
__device__ int getGlobalIdx_3D_2D()
{
  int blockId = blockIdx.x 
    + blockIdx.y * gridDim.x 
    + gridDim.x * gridDim.y * blockIdx.z; 
  int threadId = blockId * (blockDim.x * blockDim.y)
    + (threadIdx.y * blockDim.x)
    + threadIdx.x;
  return threadId;
}

//3D grid of 3D blocks
__device__ int getGlobalIdx_3D_3D()
{
  int blockId = blockIdx.x 
    + blockIdx.y * gridDim.x 
    + gridDim.x * gridDim.y * blockIdx.z; 
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
    + (threadIdx.z * (blockDim.x * blockDim.y))
    + (threadIdx.y * blockDim.x)
    + threadIdx.x;
  return threadId;
}
~~~

For example a simple program prints the thread layout from the GPU function.

~~~bash
//-*-C++-*-
#include <stdio.h>

__global__ void helloCUDA()
{
 // converting 2D thread structure into 1D thread structure 
  int i = blockIdx.x * blockDim.x * blockDim.y 
  + threadIdx.y * blockDim.x + threadIdx.x;
  printf("Hello thread %d\n", i);
  __syncthreads();       
}

int main()
{
  // Thread organization (2D block)
  dim3 comp_G(1, 1, 1);    
  dim3 comp_B(8, 8, 1); 

  helloCUDA<<<comp_G, comp_B>>>();

  cudaDeviceReset();

  return 0;
}
~~~



## Day 3

## CUDA API for C/C++

In this section, we will go through a few of the API that is available in the CUDA programming language and these will instruct the compilers to do the assigned job. 



* In this section, we will explain some of the important CUDA API that will be used for converting the C/C++ code into GPU CUDA code. 

* In CUDA GPU programming, `Host` refers to CPU, and `Device` refers to GPU.  

* The tables below show some of the commonly used function type qualifiers in CUDA 
programming. 

####  Function type qualifiers: 

| Qualifier               | Description   |
| :---| ---: |
| `__device__ `       | These functions are executed only from the device.              |
| `__global__`        | These functions are executed from the device; and it can be callable from the host       |
| `__host__`          |  These functions are executed from device; and callable from the host | 
| `__noninline__` `__forceinline__` | Compiler directives instruct the functions to be inline or not inline | 


#### More clear description:

| Qualifier                               | Executed on the: | Only callable  from the: |
|--------------------------------|------------------|--------------------------|
| `__device__`                     | Device           | Device                   |
| `__global__`                     | Device           | Host                     |
| `__host__`                       | Host             | Host                     |
| `__noninline__` `__forceinline__` | Device           | Device                   |



####  Variable types qualifier:

| Qualifier               | Description   |
| :---| ---: |
| `__device__ `       | The variables that are declared with __device__ will reside in the global memory; this means it can be accessible from the device as well as from the host (through the CUDA runtime library).              |
| `__constant__`        | It resides in the constant memory and accessible from all the threads within the grid and from the host through the runtime |
| `__shared__`          | Declared variable will be residing in the shared memory of a thread block and will be only accessible from all the threads within the block | 

#### CUDA thread qualifier:

| Qualifier               | Description   |
| :---| ---: |
| `gridDim` | type is `dim3`; size and dimension of the grid | 
| `blockDim` | type is `dim3`; block dimension in the grid| 
| `blockIdx` | type is `uint3`; block index in the blocks| 
| `threadIdx` | type `uint3`; thread index within the blocks | 
| `WrapSize` | type is `int`;  size of the warp (thread numbers)| 






## Day 4

## Vector Operations

In this section, we will study how to do the CUDA programming for the vector operations from the numerical linear algebra. 

## Basic Vector Operations 

We will not look into the basic vector operations. We will focus on the following concept in the vector addition operations:


![alt text](https://drive.google.com/uc?export=view&id=1l_i7zIXEhHwqhKuC8h9ZTw5HJ6Ll0R-l)

*Figure 1: Schematic view of the vector addition*

1. [CPU vector addition](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week2/Day4/Vector-Addition.c)
2. [GPU vector addition from 1D thread to 1D thread](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week2/Day4/Vector-Addition-Version-1.cu)
     -  [GPU vector addition from 2D thread to 1D thread](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week2/Day4/Vector-Addition-Version-2.cu)



A typical CPU vector addition function would be written as shown below, where each component of the vector is added through the `loop`. This is the well-known standard vector addition method in the CPU. Now we will focus on how to do this vector addition on the GPU (or how to convert this CPU function to GPU function). 

#### CPU vector addition 

~~~bash
// CPU function that adds two vector 
float * vector_add(float *a, float *b, float *out, int n) 
{
  for(int i = 0; i < n; i ++)
    {
      out[i] = a[i] + b[i];
    }
  return out;
}
~~~

#### GPU vector addition 

The procedure for the GPU implementation steps are as follows:

* Memory allocation on both CPU and GPU

~~~bash
  // Initialize the memory on the host
  float *a, *b, *out;

  // Allocate host memory
  a   = (float*)malloc(sizeof(float) * N);
  b   = (float*)malloc(sizeof(float) * N);
  out = (float*)malloc(sizeof(float) * N);
       
  // Initialize the memory on the device
  float *d_a, *d_b, *d_out;

  // Allocate device memory
  cudaMalloc((void**)&d_a, sizeof(float) * N);
  cudaMalloc((void**)&d_b, sizeof(float) * N);
  cudaMalloc((void**)&d_out, sizeof(float) * N); 
~~~

* Fill values for host vectors a and b. 

~~~bash
  // Initialize host arrays
  for(int i = 0; i < N; i++)
    {
      a[i] = 1.0f;
      b[i] = 2.0f;
    }
~~~

* Transfer the data from CPU to GPU.

~~~bash
  // Transfer data from host to device memory
  cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
~~~

* Thread block creation. Here initial we will get 512 threads

~~~bash
  // Thread organization 
  dim3 dimGrid(1, 1, 1);    
  dim3 dimBlock(8, 8, 1); 
~~~

* Call the CUDA kernel function 

~~~bash
// execute the CUDA kernel function 
  vector_add<<<dimGrid, dimBlock>>>(d_a, d_b, d_out, N);
~~~

* CUDA kernel vector addition function 

~~~bash
// GPU function that adds two vectors 
__global__ void vector_add(float *a, float *b, 
	   float *out, int n) 
{
	
  int i	= blockIdx.x * blockDim.x * blockDim.y + 
    threadIdx.y * blockDim.x + threadIdx.x;   
  // Allow the   threads only within the size of N
  if(i < n)
    {
      out[i] = a[i] + b[i];
    }

  // Synchronice all the threads 
  __syncthreads();
}
~~~

* Transfer data back to GPU (from device to host). Here it will be a vector that contains the value of the addition of two vectors. 

~~~bash
  // Transfer data back to host memory
  cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
~~~

* Free the host and device memory 

~~~bash
  // Deallocate device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);

  // Deallocate host memory
  free(a); 
  free(b); 
  free(out);
~~~

#### GPU vector addition from 2D thread block into 1D thread block

* Thread organization and kernel function execution 

~~~bash
  // Thread organization 
  dim3 dimGrid(1, 1, 1);    
  dim3 dimBlock(8, 8, 1); 
~~~

* 2D block of the thread is converted into 1D thread block

~~~bash
// GPU function that adds two vectors 
__global__ void vector_add(float *a, float *b, 
                                            float *out, int n) 
{
 int blockId = blockIdx.x + blockIdx.y * gridDim.x;     
 int i = blockId * (blockDim.x * blockDim.y) +          
    (threadIdx.y * blockDim.x) + threadIdx.x;
  
  // Allow the threads only within the size of N
  if(k < n)
    {
      out[i] = a[i] + b[i];
    }

  // Synchronice all the threads 
  __syncthreads();
}
~~~






## Day 5

## Matrix Operations

In this section, we will go through how to write the CUDA programming for matrix multiplication application. 

## Basic Matrix Operations 

We will now look into the basic matrix multiplication and focus on the following concepts in matrix multiplication:

1. [CPU matrix multiplication](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week2/Day5/Matrix-Multiplication.cc)
2. [GPU matrix multiplication](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week2/Day5/Matrix-Multiplication.cu) (blocked version)

#### CPU matrix multiplication

We now look into the loop for matrix multiplication. Here we notice that the matrix is stored in a 1D array because we want to consider the same function concept for CPU and GPU. 
  
~~~bash
float * matrix_mul(float *h_a, float *h_b, float *h_c, int width)   
{                                                                 
  for(int row = 0; row < width ; ++row)                           
    {                                                             
      for(int col = 0; col < width ; ++col)                       
        {                                                         
          float single_entry = 0;                                       
          for(int i = 0; i < width ; ++i)                         
            {                                                     
              single_entry += h_a[row*width+i] * h_b[i*width+col];      
            }                                                     
          h_c[row*width+col] = single_entry;                            
        }                                                         
    }   
  return h_c;           
}
~~~


#### GPU vector addition 

Now we will discuss matrix multiplication using the CUDA. The procedure for the GPU implementation steps are as follows:

* Initialize the memory allocation on both CPU and GPU

~~~bash
  // Initialize the memory on the host
  float *a, *b, *c;

  // Allocate host memory
  a = (float*)malloc(sizeof(float) * (N*N));
  b = (float*)malloc(sizeof(float) * (N*N));
  c = (float*)malloc(sizeof(float) * (N*N));
       
  // Initialize the memory on the device
  float *d_a, *d_b, *d_c;

  // Allocate device memory
  cudaMalloc((void**)&d_a, sizeof(float) * (N*N));
  cudaMalloc((void**)&d_b, sizeof(float) * (N*N));
  cudaMalloc((void**)&d_c, sizeof(float) * (N*N)); 
~~~

* Fill values for host matrix a and b. 

~~~bash
  // Initialize host arrays
  for(int i = 0; i < N; i++)
    {
      a[i] = 1.0f;
      b[i] = 2.0f;
    }
~~~

* Transfer the data from CPU to GPU.

~~~bash
  // Transfer data from host to device memory
  cudaMemcpy(d_a, a, sizeof(float) * (N*N), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(float) * (N*N), cudaMemcpyHostToDevice);
~~~

* Thread block creation and 512 threads per block is created

~~~bash
  // Thread organization
  int blockSize = 32;
  dim3 dimBlock(blockSize,blockSize,1);
  dim3 dimGrid(ceil(N/float(blockSize)),ceil(N/float(blockSize)),1);
~~~


* Call the CUDA kernel matrix multiplication function 

~~~bash
// execute the CUDA kernel function 
  matrix_mul<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, N);
~~~

* CUDA kernel matrix multiplication function 

~~~bash
__global__ void matrix_mul(float* d_a, float* d_b,
			   float* d_c, int width) 
{
  
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
    
  if ((row < width) && (col < width)) 
    {
      float single_entry = 0;
      // each thread computes one 
      // element of the block sub-matrix
      for (int i = 0; i < width; ++i) 
	{
	  single_entry += d_a[row*width+i]*d_b[i*width+col];
	}
      d_c[row*width+col] = single_entry;
    }
}
~~~

* Transfer the data back to GPU (from device to host). Here the c matrix that contains the product of the two matrices. 

~~~bash
  // Transfer data back to host memory
  cudaMemcpy(c, d_c, sizeof(float) * (N*N), cudaMemcpyDeviceToHost);
~~~

* Free the host and device memory 

~~~bash
  // Deallocate device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  // Deallocate host memory
  free(a); 
  free(b); 
  free(c);
~~~



