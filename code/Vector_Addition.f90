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

program main
  use Vector_Addition_Mod
  implicit none
  
  ! Size of vectors
  integer :: n = 100000, i
  ! Input vectors
  real(8), dimension(:), allocatable :: a
  real(8), dimension(:), allocatable :: b 
  ! Output vector
  real(8), dimension(:), allocatable :: c
  real(8) :: sum = 0
  
  ! Allocate memory for vector
  allocate(a(n))
  allocate(b(n))
  allocate(c(n))
  
  ! Initialize content of input vectors, 
  ! vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
  do i = 1, n
     a(i) = sin(i*1D0) * sin(i*1D0)
     b(i) = cos(i*1D0) * cos(i*1D0) 
  enddo
    
  ! Call the vector add subroutine 
  call Vector_Addition(a, b, c, n)

  ! Sum up vector c and print result divided by n, 
  ! this should equal 1 within error
  do i = 1, n
     sum = sum +  c(i)
  enddo
  sum = sum/n
  print *, 'final result: ', sum
  
  ! Delete the memory
  deallocate(a)
  deallocate(b)
  deallocate(c)
  
end program main
