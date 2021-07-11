## Week 5: OpenACC (advanced): Numerical Algebra, Advnaced Topics, Profiling and Tuning

This week, we will discuss how to use the OpenACC programming model for numerical algebra operations. And also we show how to use the shared memory and asynchronous concept in OpenACC. Continuing with how to profile the OpenACC code, profiling information gives an overview of how the application's function time consumption, memory transfers, and parallel strategy. Finally, we show how to optimize the code further so that it can completely utilize the parallel architecture capabilities.

## Vector Operations

In this section, we will see the simple vector addition using both C/C++ and Fortran programming language.

In this article, we will go through how to do the vector addition in C/C++ and Fortran programming language. 

#### For C/C++:
The blow listing shows the simple [Vector_Addition.c](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week5/Day1/Vector_Addition.c) function that adds the two vector into one.  

~~~cpp
// Vector_Addition.c 
float * Vector_Addition(float *a, float *b, float *c, int n) 
{
  for(int i = 0; i < n; i ++)
    {
      c[i] = a[i] + b[i];
    }
  return c;
}
~~~

Now, we need to use the OpenACC directives to make this function run in parallel. To do that, we need to add OpenACC compute directives, for example, `parallel` or `kernels`. And also, this function involves the `loop`; when there is a loop, it is better to use the `loop` clause along with the compute directive. See the below example for the OpenACC enabled [Vector_Addition_OpenACC.c](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week5/Day1/Vector_Addition_OpenACC.c) function. We can also notice that we have used the `copyin` and `copyout`; these are needed to define the data transfer between CPU and GPU. For simpler cases, the compiler will be able to fix these things automatecally even if they have not mentioned these clauses. However, to avoid any possibility of error, one should use the `copyin` and `copyout` clauses. 

~~~cpp
// Vector_Addition_OpenACC.c
void Vector_Addition(float *a, float *b, float *restrict c, int n)  
{                                                                   
#pragma acc kernels loop copyin(a[:n], b[0:n]) copyout(c[0:n])      
  for(int i = 0; i < n; i ++)                                       
    {                                                               
      c[i] = a[i] + b[i];                                           
    }                                                               
} 
~~~

> OpenACC compiler needs to avoid the pointer `aliasing` to do that; we need to use the `restrict` to the pointer variable. This is essentially needed where we need to update the array values, where the pointer points to. 

#### For Fortran:

Similarly, the below example shows the serial code function of [Vector_Addition.f90](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week5/Day1/Vector_Addition.f90) in Fortran code. 

~~~fortran
!!!Vector_Addition.f90
module Vector_Addition_Mod  
  implicit none 
contains
  subroutine Vector_Addition(a, b, c, n)
    ! Input vectors
    real(8), intent(in), dimension(:) :: a
    real(8), intent(in), dimension(:) :: b
    real(8), intent(out), dimension(:) :: c
    integer :: i, n
    do i = 1, n
       c(i) = a(i) + b(i)
    end do
  end subroutine Vector_Addition
end module Vector_Addition_Mod
~~~

> For Fortran, the index starts with `1`, not like `0` as in the C/C++ programming language. 

~~~fortran
!!! Vector_Addition_OpenACC.f90
module Vector_Addition_Mod                                                        
  implicit none                                                                   
contains                                                                          
  subroutine Vector_Addition(a, b, c, n)                                          
    ! Input vectors                                                               
    real(8), intent(in), dimension(:) :: a                                        
    real(8), intent(in), dimension(:) :: b                                        
    real(8), intent(out), dimension(:) :: c                                       
    integer :: i, n                                                               
    !$acc kernels loop copyin(a(1:n), b(1:n)) copyout(c(1:n))                     
    do i = 1, n                                                                   
       c(i) = a(i) + b(i)                                                         
    end do                                                                        
    !$acc end kernels                                                             
  end subroutine Vector_Addition                                                  
end module Vector_Addition_Mod 
~~~

> As we can notice here [Vector_Addition_OpenACC.f90](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week5/Day1/Vector_Addition_OpenACC.f90), we have been using only `kernels` compute directive. This is because the `kernels` compute directive is safer, and the compiler executes code below sequentially if there are any dependencies in the multiple loops. This has already been explained in the compute constructs section before.

## Day 2

## Matrix Operations

In this section, we will study how to do the matrix multiplication using the OpenACC for C/C++ and Fortran languages. Plus also we will use a few of the important OpenACC clauses with examples.

In this article, we study matrix multiplication in both C/C++ and Fortran languages. Plus also we will see how to use a few of the important OpenACC clauses in these examples. 

#### For C/C++
The below example shows the [Matrix_Multiplication.c](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week5/Day2/Matrix_Multiplication.c) function that multiplies the two matrices. Here we use the single array entries to represent all three matrices. And we assume that all three matrices have the same dimensional size. 

~~~cpp
/// Matrix_Multiplication.c
void Matrix_Multiplication(float *a, float *b, float *restrict c, int width)   
{ 
  float sum = 0;
  for(int row = 0; row < width ; ++row)                           
    {                                                             
      for(int col = 0; col < width ; ++col)                       
        {                                                         
          for(int i = 0; i < width ; ++i)                         
            {                                                     
              sum += a[row*width+i] * b[i*width+col];      
            }                                                     
          c[row*width+col] = sum;                           
	  sum = 0;
        }
    }   
}
~~~

The example below shows the [Matrix_Multiplication_OpenACC.c](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week5/Day2/Matrix_Multiplication_OpenACC.c) using OpenACC.

> As we can notice here, we have used the `collapse` and `reduction` clauses. This is because we are dealing with three nested loops and one reduction operation. The nested loop can be converted into just one single loop by the compiler. It is done by using the `collapse` clause. However, to be able to use the `collapse`, the loops should be tightly nested; there should not be any code between the loops. 

~~~cpp
/// Matrix_Multiplication_OpenACC.c
void Matrix_Multiplication(float *a, float *b, float *restrict c, int width)   
{ 
  int length=width*width;
  float sum = 0;
#pragma acc kernels copyin(a[0:(length)], b[0:(length)]), copyout(c[0:(length)])
#pragma acc loop collapse(3) reduction (+:sum) 
  for(int row = 0; row < width ; ++row)                           
    {                                                             
      for(int col = 0; col < width ; ++col)                       
        {                                                         
          for(int i = 0; i < width ; ++i)                         
            {                                                     
              sum += a[row*width+i] * b[i*width+col];      
            }                                                     
          c[row*width+col] = sum;
	  sum = 0;                            
        }
    }   
}
~~~

> Again, here is also, we need to use the `restrict`  for array update, which is defined as a pointer. This will prevent the pointer, no aliasing. 


#### For Fortran:

The below function shows the [Matrix_Multiplication.f90](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week5/Day2/Matrix_Multiplication.f90) in Fortran code. 

~~~fortran
!!! Matrix_Multiplication.f90
module Matrix_Multiplication_Mod  
  implicit none 
contains
  subroutine Matrix_Multiplication(a, b, c, width)
    ! Input vectors
    real(8), intent(in), dimension(:) :: a
    real(8), intent(in), dimension(:) :: b
    real(8), intent(out), dimension(:) :: c
    real(8) :: sum = 0
    integer :: i, row, col, width

    do row = 0, width-1
       do col = 0, width-1
          do i = 0, width-1
             sum = sum + (a((row*width)+i+1) * b((i*width)+col+1))
          enddo
          c(row*width+col+1) = sum
          sum = 0
       enddo
    enddo

  end subroutine Matrix_Multiplication
end module Matrix_Multiplication_Mod
~~~

The OpenACC version of [Matrix_Multiplication_OpenACC.f90](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week5/Day2/Matrix_Multiplication_OpenACC.f90) can be seen in the below example. 

~~~fortran 
!!! Matrix_Multiplication_OpenACC.f90
module Matrix_Multiplication_Mod  
  implicit none 
contains
  subroutine Matrix_Multiplication(a, b, c, width)
    ! Input vectors
    real(8), intent(in), dimension(:) :: a
    real(8), intent(in), dimension(:) :: b
    real(8), intent(out), dimension(:) :: c
    real(8) :: sum = 0
    integer :: i, row, col, width, length 
    length = width*width
    !$acc kernels copyin(a(1:length), b(1:length)) copyout(c(1:length))
    !$acc loop collapse(3) reduction(+:sum)
    do row = 0, width-1
       do col = 0, width-1        
          do i = 0, width-1
             sum = sum + (a((row*width)+i+1) * b((i*width)+col+1))
          enddo
          c(row*width+col+1) = sum
          sum = 0
       enddo
    end do
    !$acc end kernels
    
  end subroutine Matrix_Multiplication
end module Matrix_Multiplication_Mod
~~~

There is also another `OpenACC` clause that can be used when we have a nested loop with frequently used data, which is called `tile`. The below example, shows how tile can be used in the nested loop for the [Matrix_Multiplication_Tile_OpenACC.c](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week5/Day2/Matrix_Multiplication_Tile_OpenACC.c)

~~~cpp
/// Matrix_Multiplication_OpenACC.c
void Matrix_Multiplication(float *a, float *b, float *restrict c, int width)   
{ 
  int length=width*width;
  float sum = 0;
#pragma acc kernels copyin(a[0:(length)], b[0:(length)]), copyout(c[0:(length)])
#pragma acc loop tile(32,32) reduction (+:sum) 
  for(int row = 0; row < width ; ++row)                           
    {                                                             
      for(int col = 0; col < width ; ++col)                       
        {                                                         
          for(int i = 0; i < width ; ++i)                         
            {                                                     
              sum += a[row*width+i] * b[i*width+col];      
            }                                                     
          c[row*width+col] = sum;
	  sum = 0;                            
        }
    }   
}
~~~
 
And for the Fortran [Matrix_Multiplication_Tile_OpenACC.f90](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week5/Day2/Matrix_Multiplication_Tile_OpenACC.f90), please see the below example:

~~~fortran 
!!! Matrix_Multiplication_OpenACC.f90
module Matrix_Multiplication_Mod  
  implicit none 
contains
  subroutine Matrix_Multiplication(a, b, c, width)
    ! Input vectors
    real(8), intent(in), dimension(:) :: a
    real(8), intent(in), dimension(:) :: b
    real(8), intent(out), dimension(:) :: c
    real(8) :: sum = 0
    integer :: i, row, col, width, length 
    length = width*width
    !$acc kernels copyin(a(1:length), b(1:length)) copyout(c(1:length))
    !$acc loop tile(32,32) reduction(+:sum)
    do row = 0, width-1
       do col = 0, width-1        
          do i = 0, width-1
             sum = sum + (a((row*width)+i+1) * b((i*width)+col+1))
          enddo
          c(row*width+col+1) = sum
          sum = 0
       enddo
    end do
    !$acc end kernels
    
  end subroutine Matrix_Multiplication
end module Matrix_Multiplication_Mod
~~~

> In both cases, that is C/C++ and Fortran version, if we want to use the `tile` clause, we need to remove the `collapse`. 


## Day 3

## Shared Memory and Async (similar to CUDA streams)

In this section, we show how to use the shared memory from the GPUs by using the OpenACC and also how to enable Async, similar to CUDA streams in CUDA.


This article will show the usage of `cache` and `asynchronous` in OpenACC. 

OpenACC `cache` directive provides the cache clause to cache the frequently used data in the application. It is possible to cache the entire array or part of the array, however, it entirely depends on the memory of the given GPU. If the shared memory can not accommodate the allocated array, it will still load the data from the global memory, in that case, it is better to cache partial data of the array.  

![figure](https://drive.google.com/uc?export=view&id=1C2JIRhwslMU7D4Rb1R0Tiu0CYLfH2i3t)
*Figure 1: Simple example of data caching in OpenACC.* 

Figure 1 shows the example of cache data in the shared memory of the GPUs, where threads try to access a few of the array values frequently, that is, `A(i-1)`, `A(i)` and `A(i+1)`. So in this case, A of 2 values can be cached by `A(i:3)`. The following example will show the cache used in `C/C++` and `Fortran`. 

#### `cache` in `C/C++`:

~~~openmp
// function that adds two vector 
void Vector_Addition(float *a, float *b, float *restrict c, int n) 
{
#pragma acc kernels loop copyin(a[:n], b[0:n]) copyout(c[0:n])
  for(int i = 0; i < n-2; i ++)
    {
#pragma acc cache(a[i:3])
      c[i] = a[i] + a[i+1] + a[i+2] +  b[i];
    }
}
~~~

In the above example, we see that two elements of the array are being used by all threads frequently, so it can be cached, see [Vector_Cache_OpenACC.c](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week5/Day3/Vector_Cache_OpenACC.c).

#### `cache` in `Fortran`:
~~~fortran
  subroutine Vector_Addition(a, b, c, n)
    ! Input vectors
    real(8), intent(in), dimension(:) :: a
    real(8), intent(in), dimension(:) :: b
    real(8), intent(out), dimension(:) :: c
    integer :: i, n
    !$acc kernels loop copyin(a(1:n), b(1:n)) copyout(c(1:n)) 
    do i = 1, n-2
       !$acc cache(A(i:3))
       c(i) = a(i) + a(i+1) + a(i+2) + b(i)
    end do
    !$acc end kernels
  end subroutine Vector_Addition
~~~
The above example shows the earlier discussed `C/C++` example of similar array indexing of using `cache` in the Fortran language, see [Vector_Cache_OpenACC.f90](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week5/Day3/Vector_Cache_OpenACC.f90).


#### Unified Memory (Managed)
In OpenACC unified memory option is enabled by the `managed` compiler flag. When we use the unified memory, we do not need to use any typical data clauses options are in OpenACC. However, in `C/C++` if the pointer data `restrict` is used, then it should use the `independent`,. The below example, [Vector_Addition_Managed_OpenACC.c](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week5/Day3/Vector_Addition_Managed_OpenACC.c) shows how we can remove the `restrict` and include the `independent` clause in `OpenACC C/C++`.

~~~OpenMP
void Vector_Addition(float *a, float *b, float *c, int n) 
{
#pragma acc kernels loop independent 
  for(int i = 0; i < n; i ++)
    {
      c[i] = a[i] + b[i];
    }
}
~~~

![figure](https://drive.google.com/uc?export=view&id=1ieo4iCEGFYrlS1qX2JxRxE8JTkVnBZ3j)
*Figure 2: Overview of Unified memory*

Figure 2 shows the overview of the unified memory on the GPUs. To enable or tell the compiler to make it unified memory methodology, it should be compiled by the following command flag called `managed`.

~~~bash
pgcc -fast -acc -ta=tesla,managed -Minfo=all  Vector_Addition_Managed_OpenACC.c
~~~

For the `Fortran` code, it simply enough to remove the copy data clauses, the example, 
[Vector_Addition_Managed_OpenACC.f90](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week5/Day3/Vector_Addition_Managed_OpenACC.f90) shows the unified memory in the `Fortran` programming language. 

~~~fortran
  subroutine Vector_Addition(a, b, c, n)                                   
    ! Input vectors                                                        
    real(8), intent(in), dimension(:) :: a                                 
    real(8), intent(in), dimension(:) :: b                                 
    real(8), intent(out), dimension(:) :: c                                
    integer :: i, n                                                        
    !$acc kernels loop                                                     
    do i = 1, n                                                            
       c(i) = a(i) + b(i)                                                  
    end do                                                                 
    !$acc end kernels                                                      
  end subroutine Vector_Addition 
~~~

Here also it should be compiled by `managed` compiler flag to enable the unified memory concept in the application. 

~~~bash
pgfortran -fast -acc -ta=tesla,managed -Minfo=all  Vector_Addition_Managed_OpenACC.f90
~~~





## Day 4
## Profiling

In this section, we will show how to profile the OpenACC code. By doing that, we will come to know how much time each function takes. But also, time taken for the memory transfer between CPU and GPU and OpenACC APIs.

In this article, we will study how to profile both C/C++ and Fortran the code and see important traces and metrics in the application. 

Profiling is very important to analyze the code to see where it spends most of the time. This will give us detailed information about all the functions, time consumption, memory transfer, and all the API time consumption. There are few tools are exiting to  profile the OpenACC code, which is as follows:

* [Allinea, Allinea MAP](https://www.arm.com/products/development-tools/server-and-hpc/forge)
* [ScoreP, Score-P Community](https://www.vi-hps.org/projects/score-p)
* [TAU Performance System](http://www.cs.uoregon.edu/research/tau/home.php)
* [Vampir - Performance Optimization](https://vampir.eu/)
* [NVIDIA Nsight Tools](https://developer.nvidia.com/tools-overview)

Among these will go through how to use the [PGI compiler](https://www.pgroup.com/index.htm) for the profiling. The PGI profiling is already a part of the [Nvidia HPC SDK](https://developer.nvidia.com/hpc-sdk). 

PGI compiler for the OpenACC provides the parallel strategy and data movement information at the compile time. This applies to both GPUs and CPUs. 
 
To see the whole code profiling information, please use:

~~~bash
pgcc -fast -Minfo=all -ta=tesla -acc Vector_Addition_OpenACC.c
pgfortran -fast -Minfo=all -ta=tesla -acc Vector_Addition_OpenACC.f90
~~~

To see just kernel profiling information, please use:

~~~bash
pgcc -fast -Minfo=accel -ta=tesla -acc Vector_Addition_OpenACC.c
pgfortran -fast -Minfo=accel -ta=tesla -acc Vector_Addition_OpenACC.f90
~~~

#### Command line profiling:

The following steps will provide a detailed view of the profiling step by step:

- The first step would be just to compile the entire code: 

      pgcc -fast -Minfo=all -ta=tesla -acc Vector_Addition_OpenACC.c

- Then, if you do not know what to look for in the profiling, then please type the following command to query the [list of options](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week5/Day4/pgprof.txt):

      // this will show the list of options that `pgprof` provides. 
      pgprof --help 

- For example, to see the following information:
   -  GPU kernel execution profile
   
          pgprof --print-gpu-summary ./a.out
          pgprof --print-gpu-trace ./a.out

   -  CUDA API execution profile
 
          pgprof --print-api-summary ./a.out
          pgprof --print-api-trace ./a.out

   -  OpenACC execution profile

          pgprof --print-openacc-trace ./a.out
          pgprof --print-openacc-summary ./a.out

   -  CPU execution profile

          pgprof --cpu-profiling-mode flat ./a.out

#### Visual Profiling:

Sometimes we also would like to see the visual profiler, especially to see the communication and computation time in the application. Because most of the time, those are the parameters we should be looking at and try to optimize the time consumption. Please refer to the below steps on how to visualize the profiled data using the `pgprof`. 

* We need to create an output file that can be opened by the `pgprof`:
          
          pgprof -o profiled-output.pgprof --cpu-profiling-mode flat ./a.out

* Then, to open the file, we need to open the GPU of `pgprof`. 
* Once the `pgprof` is opened, we can easily open the `profiled-output.pgprof` file. 
* Figure 1 shows the example of `pgprof GUI`.

![figure](https://drive.google.com/uc?export=view&id=19GzRfEO4e6wxu860n0RNWUwqRcVCKLAO)
*Figure 1: Example of pgprof GUI profiling*


> There are a few important [environmental variables](https://www.pgroup.com/resources/docs/20.4/x86/pgi-user-guide/index.htm#env-vars) which are supported by the PGI compiler and these can  be set the compilation time:

* PGI_ACC_DEBUG
  - runtime debugging
* ACC_NOTIFY
  - writes out a line for each kernel and data movement 
  - options: 1 - kernels launch; 2 - data transfer; 4 - synchronous operations; 8 - region entry/exit; 16 - data allocation/free 
* PGI_ACC_TIME
  - lightweight profiler for a summary of the program
* PGI_ACC_SYNCHRONOUS
  - disabling the synchronous operations 
* Example usage:
  - for csh: setenv ACC_NOTIFY 1
  - for bash : export ACC_NOTIFY=1




## Day 5
## Tuning and Optimization

As you all now might have familiar with OpenACC and would have noticed that, it is easy to the parallelise the serial code into OpenACC. But most of the tricks involving with the tuning and code optimization.

In this article, we will study how to optimize the OpenACC code.

OpenACC supports the gang, worker, and vector, which are similar to CUDAs, thread block, warp, and thread. This will provide good occupancy to the GPUs. However, OpeACC supports both GPUs and CPUs, so the concept of the gang, worker, and vector are varying depending on the computing architecture. Below Table 1 provides a quick overview of the functionality of the gang, worker, and vector in OpenACC.


|Platform | Gang|Worker|Vector|
|--|--|--|--|											
|Multicore	CPU|	Whole	CPU	(NUMA	domain)	|Core	|SIMD	vector|					
|Manycore	CPU	(e.g.,	Xeon	Phi)|	NUMA	domain	(whole	chip	or	quadrant)|	Core|	SIMD	vector|
|NVIDIA	GPU	(CUDA)|	Thread	block|	Warp|	Thread|							
|AMD	GPU	(OpenCL)|	Workgroup|	Wavefront|	Thread|			
					
*Table 1: Overview of different architecture and functionality of gang, worker, and vector*

Furthermore, OpenACC provides the two primary compute kernels, as we have discussed earlier. At the same time, the clauses are also varying for these two kernels, which can be seen in Table 2.

|            |OpenACC Kernels |OpenACC Parallel
|--|--|--|
|Threads| vector(expression) |vector_length(expression)|
|Warps|worker(expression) |num_workers(expression)|
|Thread Block|gang(expression) |num_gangs(expression)|
|Device (nvidia or radeon) |device_type(device name) |device_type(device name)|

*Table 2: CLAUSES for parallel and kernels*

Now let's consider a simple example, where we can optimize the thread blocks in the computation. In the `CUDA` we specifically mention, how is our thread blocks should be executed. Whereas on the `OpenACC` we can not provide like in `CUDA`. But, we can use the clauses to control the thread blocks in the `OpenACC`. By default, the `OpenACC` compiler chooses the best combination of the threads blocks depends on the architecture. However, sometimes, this option does not work depends on the problem you are trying to compute. The below example shows the simple case of default threads creation by the compiler. 

### Gang, Worker, Vector Clauses:

~~~openmp
// function that adds two vector 
void Vector_Addition(float *a, float *b, float *restrict c, int n) 
{
#pragma acc kernels loop copyin(a[:n], b[0:n]) copyout(c[0:n])
  for(int i = 0; i < n; i ++)
    {
      c[i] = a[i] + b[i];
    }
}

###### Profiling Output ######
Vector_Addition:
     12, Generating copyin(a[:n]) [if not already present]
         Generating copyout(c[:n]) [if not already present]
         Generating copyin(b[:n]) [if not already present]
     13, Loop is parallelizable
         Generating Tesla code
         13, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
~~~

Now, assume that you are not happy with the thread blocks due to the bad performance, and you want to have your own optimized threads blocks. To do that we can use the `OpenACC` clauses, as we can see in the below example. 

~~~openmp
// function that adds two vector 
void Vector_Addition(float *a, float *b, float *restrict c, int n) 
{
#pragma acc kernels loop gang(5) worker(32) vector(32) copyin(a[:n], b[0:n]) copyout(c[0:n])
  for(int i = 0; i < n; i ++)
    {
      c[i] = a[i] + b[i];
    }
}
//////////// or /////////////////////////
// function that adds two vector 
void Vector_Addition(float *a, float *b, float *restrict c, int n) 
{
#pragma acc kernels copyin(a[:n], b[0:n]) copyout(c[0:n])
#pragma acc loop gang(5) worker(32) vector(32)
  for(int i = 0; i < n; i ++)
    {
      c[i] = a[i] + b[i];
    }
}

The below example shows the compilation output from the including `OpenACC` clauses. As we can see here, the loop is thread blocked by the `thread blocks`, `warps`, and `threads`. 

###### Profiling Output ######
Vector_Addition:
     12, Generating copyin(a[:n]) [if not already present]
         Generating copyout(c[:n]) [if not already present]
         Generating copyin(b[:n]) [if not already present]
     13, Loop is parallelizable
         Generating Tesla code
         13, #pragma acc loop gang(5), worker(32), vector(32) /* blockIdx.x threadIdx.y threadIdx.x */
~~~

The following example is similar to one that shown earlier but with `acc parallel`. 

With default threads:

~~~openmp
// function that adds two vector 
void Vector_Addition(float *a, float *b, float *restrict c, int n) 
{
#pragma acc parallel loop copyin(a[:n], b[0:n]) copyout(c[0:n])
  for(int i = 0; i < n; i ++)
    {
      c[i] = a[i] + b[i];
    }
}

###### Profiling Output ######
Vector_Addition:
     15, Generating copyin(a[:n]) [if not already present]
         Generating copyout(c[:n]) [if not already present]
         Generating copyin(b[:n]) [if not already present]
         Generating Tesla code
         18, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
~~~

 With using `num_gangs()`, `num_workers()` and `vector_length()`:

~~~openmp
// function that adds two vector 
void Vector_Addition(float *a, float *b, float *restrict c, int n) 
{
#pragma acc parallel loop num_gangs(5) num_workers(32) vector_length(32) copyin(a[:n], b[0:n]) copyout(c[0:n])
  for(int i = 0; i < n; i ++)
    {
      c[i] = a[i] + b[i];
    }
}

###### Profiling Output ######
Vector_Addition:
     16, Generating copyin(a[:n]) [if not already present]
         Generating copyout(c[:n]) [if not already present]
         Generating copyin(b[:n]) [if not already present]
         Generating Tesla code
         18, #pragma acc loop gang(5), worker(32), vector(32) /* blockIdx.x threadIdx.y threadIdx.x */
~~~
