# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: venv (3.12.3)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NLP interface & Solver basics
#
# THis tutorial illustrates the generic NLP interface, access to a solver, and plotting the traces.

# %%
import optsam as op
import numpy as np
import matplotlib.pyplot as plt


# %% [markdown]
# Define a problem, here: 2 dimensional, with 2 SOS features, and bounds [-1,-2]x[2,3]

# %%
class MyNLP:
    def __init__(self):
        self.dimension = 2
        self.types = [op.OT.sos] * 2
        self.bounds = np.array([[-2,-1], [2,3]])
        
        self.b = 3

    def evaluate(self, x):
        phi = np.array([ x[0]-1,
                         self.b*(x[1]-x[0]**2) ])
        J = np.array([[ 1, 0 ],
                      [ -2*self.b*x[0], self.b ]])
        return phi, J
    
nlp = MyNLP()


# %% [markdown]
# Create a solver

# %%
sol = op.NLP_Solver()
sol.setPyProblem(nlp)
sol.setOptions(stepMax=.5, damping=1e-4)
sol.setTracing(trace_x=True, trace_errs=True)

# %% [markdown]
# Call the solver. Here 20 times in a row, each time automatically initialized with uniform in the bounds (default implementation of nlp.getInitializationSample)

# %%
trace_x = []
trace_err = []

for i in range(20):
    ret = sol.solve(1)
    print(ret)

    trace_x.append(sol.getTrace_x())
    trace_err.append(sol.getTrace_errs())
    sol.clearTracing()

trace_f = [np.sum(E, axis=1) for E in trace_err]

# %% [markdown]
# The following creates a grid X of input points, and evaluates the fct on X

# %%
nlp = sol.getProblem()
B = nlp.bounds
X = [None] * nlp.dimension
for i in range(nlp.dimension):
    X[i] = np.linspace(B[0][i], B[1][i], 30)
X = np.stack(np.meshgrid(*X, indexing='ij'), axis=-1) 
fX = np.array([nlp.eval_scalar(x)[0] for x in X.reshape(-1, nlp.dimension)])
fX = fX.reshape(X.shape[:-1])

# %% [markdown]
# ... to prepare plotting.

# %%
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
ax1.contour(X[:,:,0], X[:,:,1], fX, 200)
for x in trace_x:
    ax1.plot(x[:,0], x[:,1], 'o-r', ms=3)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_wireframe(X[:,:,0], X[:,:,1], fX)
for x,f in zip(trace_x, trace_f):
    ax2.plot(x[:,0], x[:,1], f, 'o-r', ms=3)

plt.show()

# %% [markdown]
# Finally, an example to check the derivatives (Jacobian of all problem features) at random initialization points:

# %%
for i in range(20):
    x = sol.getProblem().getInitializationSample()
    r = sol.getProblem().checkJacobian(x, 1e-6)
    # print(r, x)

# %%
