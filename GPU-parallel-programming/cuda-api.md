* In this section, we will explain some of the important CUDA API that will be used for converting the C/C++ code into GPU CUDA code. 

* In CUDA GPU programming, `Host` refers to CPU, and `Device` refers to GPU.  

* The tables below show some of the commonly used function type qualifiers in CUDA 
programming. 

####  Function type qualifiers: 

| Qualifier               | Description   |
|-------|------------------------|
| `__device__ `       | These functions are executed only from device.              |
| `__global__`        | These functions are executed form the device; and it can be callable from the host       |
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
|---| ----|
| `__device__ `       | The variables that are declared with __device__ will reside in the global memory; this means it can be accessible from the device as well as from the host (through the CUDA run time library).              |
| `__constant__`        | It resides in the constant memory and accessible from all the threads within the grid and from the host through the runtime |
| `__shared__`          | Declared variable will be residing in the shared memory of a thread block and will be only accessible from all the threads within the block | 

#### CUDA thread qualifier:

| Qualifier               | Description   |
| ----| -----|
| `gridDim` | type is `dim3`; size and dimension of the grid | 
| `blockDim` | type is `dim3`; block dimension in the grid| 
| `blockIdx` | type is `uint3`; block index in the blocks| 
| `threadIdx` | type `uint3`; thread index within the blocks | 
| `WrapSize` | type is `int`;  size of the warp (thread numbers)| 

