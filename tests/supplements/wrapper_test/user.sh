#!/bin/sh
x1=$2
x2=$4

python -c "print(f'{($x1 ** 2) - (4.0 * $x1) + ($x2 ** 2) - $x2 - ($x1 * $x2)}')"
