# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # Test Problems
#
# This script loops through all test problems, displays their signature, cost function, and -- if robotics problem -- komo scene.

# %%
import optsam as op
import numpy as np
import matplotlib.pylab as plt
import time

def scalar_objective(nlp: op.NLP, x):
    phi, _ = nlp.evaluate(x)
    ty = nlp.getTypes()

    if len(phi) != len(ty):
        return nlp.eval_scalar(x)[0]

    cost = 0.0
    for value, feature_type in zip(phi, ty):
        if feature_type == op.OT.f:
            cost += value
        elif feature_type == op.OT.sos:
            cost += value * value
    return cost

# %% [markdown]
#
# The following is a plotting helper problem: It creates a 2D grid; if the problem is 2D it evaluates on that grid; otherwise on a random hyperplane (determined by x0). Then plots.

# %%
def display_2d_unconstrained(nlp: op.NLP, p_name, resolution=30):
    B = nlp.bounds
    x0 = nlp.getInitializationSample()

    dim = 2
    X = [None] * dim
    for i in range(dim):
        X[i] = np.linspace(B[0][i], B[1][i], resolution)
    X = np.stack(np.meshgrid(*X, indexing='ij'), axis=-1) 

    if nlp.dimension==2:
        fX = np.array([scalar_objective(nlp, x) for x in X.reshape(-1, dim)])
    else:
        f = lambda x: scalar_objective(nlp, np.concatenate((x, x0[2:])))
        fX = np.array([f(x) for x in X.reshape(-1, dim)])
    fX = fX.reshape(X.shape[:-1])

    fig, ax = plt.subplots()
    ax.contour(X[:,:,0], X[:,:,1], fX, 200)
    # ax.plot(x[:,0], x[:,1], 'o-r', ms=3)
    ax.set_title(p_name)
    plt.show()

# %% [markdown]
#
# The following displays a problem more generically: If it has constraints, it is converted to unconstraint (using the Augmented Lagrangian). If it is a robotics problem, it also displays the komo scene.

# %%
def display_any(nlp: op.NLP, p_name):
    ty = nlp.types

    komo = nlp.as_KOMO()
    if komo is not None:
        for _ in range(5):
            # x = nlp.getUniformSample()
            x = nlp.getInitializationSample()
            nlp.evaluate(x)
            komo.view(False, 'random init')
            time.sleep(.2)

    if (op.OT.eq in ty) or (op.OT.ineq in ty):
        nlp_org = nlp
        nlp = nlp.aug_lag(1e1, -1.)

    display_2d_unconstrained(nlp, p_name)

# %% [markdown]
#
# We can now loop through all test problems, print their signatur, and display.

# %%
def main():
    problems = op.get_NLP_Problem_names() # some pre-defined benchmark problems
    print(problems)

    # p = problems[5]
    for p in problems:
        nlp = op.make_NLP_Problem(p)
        print('===\n', p, nlp.report(1), '===')
        display_any(nlp, p)

# 
if __name__ == "__main__":
    main()
