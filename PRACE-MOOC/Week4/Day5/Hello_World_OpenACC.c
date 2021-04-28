// hello-world.c
#include <stdio.h>              
#include <openacc.h>

void Print_Hello_World()    
{
#pragma acc kernels
  for(int i = 0; i < 5; i++)
    {                                
      printf("Hello World!\n");
    }
} 

int main()
{ 
  Print_Hello_World();     
  return 0;     
}
