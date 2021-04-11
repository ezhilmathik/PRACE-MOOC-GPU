// hello-world.c
#include <stdio.h>              
#include <openacc.h>

void Print_Hello_World()    
{
  int i = 0, N = 5;    
#pragma acc kernels
  for(i = 0; i < N; i++)
    {                                
      printf("Hello World!\n");
    }
} 

int main()
{ 
  Print_Hello_World();     
  return 0;     
}
