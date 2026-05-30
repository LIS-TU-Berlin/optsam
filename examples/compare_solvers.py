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

# %%
# # Solvers
#
# This script runs several solvers on several test problems, displaying their performance.

import optsam as op
import numpy as np
import numpy.typing as npt
import matplotlib.pylab as plt
import time

method_info = {
    'none': (False), 
    'GradientDescent': (False), 'Rprop': (False), 'LBFGS': (False), 'Newton': (False), 
    'AugmentedLag': (True), 'LogBarrier': (True), 'slackGN_logBarrier': (True), 'SquaredPenalty': (True), 'singleSquaredPenalty': (True), 
    'slackGN': (True), 
    'NLopt': (True), 'Ipopt': (True), 'slackGN_Ipopt': (True), 'Ceres': (True), 
    'LSBO': (False), 'greedy': (False), 'NelderMead': (False), 
    'CMA': (False), 'LS_CMA': (False), 'ES': (False), 
}


def run(nlp: op.NLP, method: op.OptMethod, fixed_x0: npt.NDArray[np.float64]):
    sol = op.NLP_Solver()
    sol.setProblem(nlp)
    sol.setOptions(method=method, stopTolerance=1e-4, damping=1e-3)
    sol.setInitialization(fixed_x0)
    ret = sol.solve()
    print(f'-- method {method}: {ret}')
    return sol.getTrace_best()

def main():
    problems = op.get_NLP_Problem_names()
    print('-- all problems:', problems)
    
    # problems = ['square', 'Rugged', 'Rastrigin', 'Rosenbrock', 'Ackley', 'Himmelblau', 'Box', 'Modes', 'Wedge', 'HalfCircle', 'LinearProgram', 'IK', 'IKobstacle', 'IKtorus', 'PushToReach', 'StableSphere', 'SpherePacking', 'MinimalConvexCore']
    problems = ['square',  'Modes', 'LinearProgram', 'IK', 'IKobstacle', 'SpherePacking', 'MinimalConvexCore']
    methods = [op.OptMethod.AugmentedLag, op.OptMethod.LBFGS, op.OptMethod.Rprop, op.OptMethod.Newton ]
    probs = problems

    traces = dict()

    seeds = 2

    n = len(probs)
    fig, axes = plt.subplots(4, (n+3)//4, figsize=(20,15))

    for ax, p in zip(axes.reshape(-1), probs):
        nlp = op.make_NLP_Problem(p)
        ty = nlp.getTypes()
        is_constrained = (op.OT.eq in ty) or (op.OT.ineq in ty)
            
        print('===', p, '===')
        print(nlp.report(1))
        print('=============')
        
        for m in methods:
            traces[m.name] = []

        for s in range(seeds):
            print('--', s)
            x0 = nlp.getInitializationSample()
            for m in methods:
                is_constrained_method = method_info[m.name]
                print('--', m)
                if is_constrained and not is_constrained_method:
                    nlp_tmp = nlp.aug_lag(muSquaredPenalty=1e2)
                else:
                    nlp_tmp = nlp
                best_trace = run(nlp_tmp, m, x0)
                traces[m.name].append(best_trace)

        print('-- plot')
        ax.set_title(f'{p} ({nlp.dimension}D)')
        color = 0
        x_min = 1
        for m in methods:
            for i,t in enumerate(traces[m.name]):
                ax.plot(t, color=f'C{color}', label=(str(m.name) if i==0 else None))
                x_min = min(x_min, np.min(t))
            color += 1
        if x_min>1e-10:
            ax.set_yscale('log')

        ax.legend(loc="upper right")

        del nlp

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
