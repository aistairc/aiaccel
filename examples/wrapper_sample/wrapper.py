from aiaccel.util import aiaccel


def main():
    # Create a wrapper
    run = aiaccel.Run()

    # Optimization
    run.execute_and_report("sh user.sh", y_data_type="float")


if __name__ == "__main__":
    main()


"""
[NOTE]
- Modify your program so that the output of the objective function
is output to standard output in the form "objective_y:$value".

- If you want to run wrapper.py alone without aiaccel,
specify the command line arguments as follows

> python wrapper.py --x1 0 --x2 1
"""
