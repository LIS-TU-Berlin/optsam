---
hide-toc: true
firstpage:
lastpage:
---

```{toctree}
:hidden:
:caption: Tutorials
tutorials/nlp_interface
tutorials/test_problems
tutorials/compare_solvers
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/LIS-TU-Berlin/optsam>
```

# OptSam - Optimization and Sampling Library

A package defining an NLP abstraction, including constrained sparse NLP solvers, NLP sampling, zero-order methods, and test problems.

## Installation:

      pip install optsam

You might also need to install basic Ubuntu packages (dependencies via the robotics problems):

      sudo apt install liblapack3 freeglut3-dev libglu1-mesa libxrandr2 libfreetype6 fonts-ubuntu python3 python3-pip
      #in latest Ubuntu also:
	  cd /usr/lib/x86_64-linux-gnu/ && sudo ln -s libglut.so.3.12 libglut.so.3

## Example:

```{code-block} python

import optsam as op
import numpy as np
import matplotlib.pylab as plt

# Define a SumOfSqr over 3 dimensions
class SimpleSumOfSqr:
    def __init__(self):
        self.dimension = 3
        self.featureTypes = [op.OT.sos] * 3
        self.bounds = np.array([[-2,-2,-2],[2,2,2]])

    def evaluate(self, x):
        phi = x.copy()
        phi[0] = phi[0] - 1.
        J = np.eye(phi.size)
        return phi, J
    
    def getInitializationSample(self):
        return np.random.uniform(self.bounds[0], self.bounds[1])

nlp = SimpleSumOfSqr()

# Define a solver
solver = op.NLP_Solver()
solver.setPyProblem(nlp)
solver.setSolver(op.OptMethod.Newton)

# Run the solver
solver.solve()
solver.getTrace_best()
```
