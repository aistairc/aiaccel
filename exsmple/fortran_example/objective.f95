program objective
    implicit none
    real :: arg1, arg2, result
    character(len=32) :: arg1_str, arg2_str

    ! Get the input arguments from the command line
    call get_command_argument(1, arg1_str)
    call get_command_argument(2, arg2_str)

    ! Convert the input arguments from string to real
    read(arg1_str, *) arg1
    read(arg2_str, *) arg2

    ! Call the function to optimize
    result = function_to_optimize(arg1, arg2)
    write(*, '(F0.16)') result

contains

    real function function_to_optimize(x1, x2)
        real, intent(in) :: x1, x2
        function_to_optimize = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
    end function function_to_optimize

end program objective
