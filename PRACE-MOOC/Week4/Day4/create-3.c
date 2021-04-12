#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <openacc.h>

#define MAX_ERR 1e-6

// function that adds two vector 
void Vector_Addition(float *a, float *b, float *restrict c, int n) 
{
#pragma acc kernels loop 
    for(int i = 0; i < n; i ++)
      {
	c[i] = a[i] + b[i];
      }
}

int main()
{
  printf("This program does the addition of two vectors \n");
  printf("Please specify the vector size = ");
  int N;
  scanf("%d",&N);

  // Initialize the memory on the host
  float *a, *b, *c;       
  
  // Allocate host memory
  a = (float*)malloc(sizeof(float) * N);
  b = (float*)malloc(sizeof(float) * N);
  c = (float*)malloc(sizeof(float) * N);
  
  // Initialize host arrays
  for(int i = 0; i < N; i++)
    {
      a[i] = 1.0f;
      b[i] = 2.0f;
      c[i] = 3.0f;
    }
  
#pragma acc enter data copyin(a[0:N], b[0:N]) create(c[0:N])  
  // Executing Vector Addition funtion 
  // Vector_Addition(a, b, c, N);
#pragma acc kernels loop 
  for(int i = 0; i < N; i ++)
    {
      c[i] = a[i] + b[i];
    }
#pragma acc exit data copyout(c[0:N])  delete(a,b)
  
  
  // Verification
  for(int i = 0; i < N; i++)
    {
      assert(fabs(c[i] - 3.00) < MAX_ERR);
    }
  printf("PASSED\n");
    

 

  // Deallocate host memory
  free(a); 
  free(b); 
  free(c);

  return 0;
}
