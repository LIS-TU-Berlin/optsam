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

# # Solvers
#
# This script runs several solvers on several test problems, displaying their performance.

import optsam as op
import numpy as np
import numpy.typing as npt
import matplotlib.pylab as plt
import time

def run(nlp: op.NLP, method: op.OptMethod, fixed_x0: npt.NDArray[np.float64]):
    sol = op.NLP_Solver()
    sol.setProblem(nlp)
    sol.setOptions(method=method, stopTolerance=1e-4, damping=1e-3)
    sol.setInitialization(fixed_x0)
    ret = sol.solve()
    print(f'== method {method}: {ret}')
    best_trace = sol.getTrace_best()

    return best_trace

def plot(ax, traces, title):
    for method, best_trace in traces:
        ax.plot(best_trace, label=str(method))
    ax.legend(loc="upper right")
    ax.set_title(title)

def main():
    problems = op.get_NLP_Problem_names()
    print(problems)

    methods = [op.OptMethod.Rprop, op.OptMethod.LBFGS, op.OptMethod.Newton]
    probs = problems[:3]

    seeds = 1

    fig, axes = plt.subplots(len(probs), 1, squeeze=False)

    for ax, p in zip(axes[:,0], probs):
        nlp = op.make_NLP_Problem(p)
        print('===\n', p, nlp.report(1), '===')
        traces = []
        for s in range(seeds):
            x0 = nlp.getInitializationSample()
            for m in methods:
                best_trace = run(nlp, m, x0)
                traces.append((m, best_trace))
        plot(ax, traces, f'{p} ({nlp.dimension}D)')

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
