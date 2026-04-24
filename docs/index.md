---
hide-toc: true
firstpage:
lastpage:
---

```{toctree}
:hidden:
:titlesonly:
:caption: Introduction
content/installation
```

```{toctree}
:hidden:
:caption: Tutorials
tutorials/nlp_interface
tutorials/test_problems
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/LIS-TU-Berlin/optsam>
```

```{project-logo} _static/img/LIS-26-cube.png
:alt: LIS lab logo
```

```{project-heading}
Optimization and Sampling Library - including constrained sparse NLP solvers, NLP sampling, zero-order methods, and test problems
```

Description

```{code-block} python

import optsam as opt
import numpy as np
import matplotlib.pylab as plt

# Define a SumOfSqr over 3 dimensions
class SimpleSumOfSqr:
    def __init__(self):
        self.dimension = 3
        self.featureTypes = [opt.OT.sos] * 3
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
solver = opt.NLP_Solver()
solver.setPyProblem(nlp)
solver.setSolver(opt.OptMethod.Newton)

# Run the solver
solver.solve()
solver.getTrace_best()
```
