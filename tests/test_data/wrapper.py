from aiaccel.util import aiaccel

# Create a wrapper
run = aiaccel.Run()

# Optimization
y = run.execute_and_report("python original_main.py")
ys = run.execute("python original_main.py")
run.report(ys[0])
print(run.ys)
