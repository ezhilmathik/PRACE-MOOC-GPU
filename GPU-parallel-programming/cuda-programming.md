### Printing out _Hello World_ form GPU

* Hello world program will just demonstrate how to print the _Hello World_ from the CUDA Programming. The below code shows the simple **C programming** and simple **CUDA programming** to show the simple difference between **CPU** and **GPU** programming. 

### C
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

### CUDA
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

### Compilation instruction

~~~bash
# Load the CUDA module (compiler)         # Load the module
$ nvcc -arch=compute_70 hello-world.cu    # Compilation
$ ./a.out                                 # Code execution
~~~

* Here `-arch=compute_70` is a compute architecture for the Nvidia Volta GPUs. Each Nvidia GPUs falls into certain [compute capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities). And this can be defined by using `-arch=compute_XX`

### The following steps will demonstrate the necessary understanding:
* The main idea of using `GPU` is that we want to run our sub-task functions in the `GPU`. To do that the function should be declared as `__global__`, this means that the declared function will always run on the `GPUs`.
* And the same time, it should be also called with threads `<<<-,-,->>>`, from the main programme where it is being called. For example, from the above example it should defined as `cuda_function<<<1,1>>>()` from just as a `c_function()`. 
* Above all, we also need to synchronize the calls (both threads and device calls). Otherwise, we will get the wrong solution in the computation. Please check by your self by removing the call `cudaDeviceSynchronize();`, you will not be able to print out the `Hello World`. It is mainly due to `master thread` does not wait for the `slave thread`, in this case, you produce the wrong result in the computation.         