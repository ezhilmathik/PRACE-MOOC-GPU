program main
  use openacc
  implicit none
  integer :: i
  !$acc kernels loop
  do i = 1, 5
     print *, "hello world"
  end do
  !$acc end kernels
end program main


