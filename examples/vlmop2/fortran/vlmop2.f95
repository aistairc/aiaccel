
program vlmop2
    implicit none
    integer, parameter :: n = 10

    character(len=20) :: arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10
    real(kind=8) :: x1, x2, x3, x4, x5, x6, x7, x8, x9, x10
    REAL(KIND=8) :: x(n)
    real(kind=8) :: y1, y2

    ! read command line arguments
    call get_command_argument(1, arg1)
    call get_command_argument(2, arg2)
    call get_command_argument(3, arg3)
    call get_command_argument(4, arg4)
    call get_command_argument(5, arg5)
    call get_command_argument(6, arg6)
    call get_command_argument(7, arg7)
    call get_command_argument(8, arg8)
    call get_command_argument(9, arg9)
    call get_command_argument(10, arg10)

    read(arg1, *) x1
    read(arg2, *) x2
    read(arg3, *) x3
    read(arg4, *) x4
    read(arg5, *) x5
    read(arg6, *) x6
    read(arg7, *) x7
    read(arg8, *) x8
    read(arg9, *) x9
    read(arg10, *) x10

    x = (/x1, x2, x3, x4, x5, x6, x7, x8, x9, x10/)

    y1 = 1.0 - EXP(-SUM((x - 1.0 / SQRT(REAL(n, KIND=8)))**2))
    y2 = 1.0 - EXP(-SUM((x + 1.0 / SQRT(REAL(n, KIND=8)))**2))

    PRINT *, y1
    PRINT *, y2

end program vlmop2
