from aiaccel.util.aiaccel import Wrapper
from aiaccel.easy_visualizer import EasyVisualizer

# Set User's program, and Config file(json)
wrp = Wrapper()

# Optimization
y = wrp.doit()

# save resulti
wrp.toOptimizer(float(y[0]))
