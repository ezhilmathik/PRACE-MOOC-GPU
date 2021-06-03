#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#define N 5120
#define MAX_ERR 1e-6

// CPU function that adds two vector 
float * Vector_Add(float *a, float *b, float *out, int n) 
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
  Vector_Add(a, b, out, N);

  // Stop measuring time and calculate the elapsed time
  clock_t end = clock();
  double elapsed = (double)(end - start)/CLOCKS_PER_SEC;
        
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
