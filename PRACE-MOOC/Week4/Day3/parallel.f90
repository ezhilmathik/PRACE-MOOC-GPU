program main
  use openacc
  implicit none
  integer :: i
  !$acc parallel
  do i = 1, 5
     print *, "hello world"
  end do
  !$acc end parallel
end program main


