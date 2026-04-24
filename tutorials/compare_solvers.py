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
# ---

# %% [markdown]
# # Solvers
#
# This script runs several solvers on several test problems, displaying their performance.

# %%
import optsam as op
import numpy as np
import matplotlib.pylab as plt
import time

# %%
def run(nlp: op.NLP, method: op.OptMethod, fixed_x0: np.array):
    sol = op.NLP_Solver()
    sol.setProblem(nlp)
    sol.setOptions(method=method, stopTolerance=1e-4, damping=1e-3)
    sol.setInitialization(fixed_x0)
    ret = sol.solve()
    print(f'== method {method}: {ret}')
    errs = sol.getTrace_errs()

    return errs

# %%
def plot(errs, title):
    fig, ax = plt.subplots()
    # ax.contour(X,Y,Z, 200)
    if np.sum(errs[-1,:])<1e-3:
        ax.set_yscale('log')
    ax.plot(errs, label=['f', 'sos', 'eq', 'ineq'])
    ax.legend(loc="upper right")
    ax.set_title(title)
    plt.show()

# %%
def main():
    problems = op.get_NLP_Problem_names()
    print(problems)

    methods = [op.OptMethod.Rprop, op.OptMethod.LBFGS, op.OptMethod.Newton]
    probs = problems[:3]

    seeds = 1
    
    for p in probs:
        nlp = op.make_NLP_Problem(p)
        print('===\n', p, nlp.report(1), '===')
        for s in range(seeds):
            x0 = nlp.getInitializationSample()
            for m in methods:
                errs = run(nlp, m, x0)
                plot(errs, f'{p} ({nlp.dimension}D) {m}')

# %%
if __name__ == "__main__":
    main()

# %%
