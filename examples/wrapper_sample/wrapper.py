from aiaccel.util import aiaccel

# Create a wrapper
run = aiaccel.Run()

# Optimization
y = run.execute_and_report("python user.py")
print(run.ys)
"""
[Note]
`command` refers to the terminal command.
Command line arguments will be generated automatically.
"""
