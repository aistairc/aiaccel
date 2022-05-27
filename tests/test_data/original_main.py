import argparse
import numpy as np
import random
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x1", type=float, default=0.0)
    parser.add_argument("-x2", type=float, default=0.0)
    parser.add_argument("-x3", type=float, default=0.0)
    parser.add_argument("-x4", type=float, default=0.0)
    parser.add_argument("-x5", type=float, default=0.0)
    parser.add_argument("-x6", type=float, default=0.0)
    parser.add_argument("-x7", type=float, default=0.0)
    parser.add_argument("-x8", type=float, default=0.0)
    parser.add_argument("-x9", type=float, default=0.0)
    parser.add_argument("-x10", type=float, default=0.0)
    args = parser.parse_args()

    # Read parameters
    x1 = args.x1
    x2 = args.x2
    x3 = args.x3
    x4 = args.x4
    x5 = args.x5
    x6 = args.x6
    x7 = args.x7
    x8 = args.x8
    x9 = args.x9
    x10 = args.x10

    x = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])

    # Sphere
    y = np.sum(x ** 2)
    print("Formula: {}, input: x1 {}, x2 {}, x3 {}, x4 {}, x5 {}, x6 {},"
          " x7 {}, x8 {}, x9 {}, x10 {}, output: {}"
          .format("y = x ** 2", x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y))

    # Randomly sleep
    time.sleep(random.uniform(5.0, 10.0))

    # Output the metrics
    print(y)


if __name__ == "__main__":
    main()
