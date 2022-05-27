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


If you want to specify arguments manually,
you can describe it like below.
"""
# run = aiaccel.Run()
# p = run.parameters
# command = "python user.py --x1 {x1} --x2 {x2}".format(x1=p['x1'], x2=p['x2'])
# run.execute_and_reoprt(command, False)
