import optsam as op
import numpy as np
import matplotlib.pylab as plt
import data_tools.plot_helper
import time

def display_2d_unconstrained(nlp: op.NLP, resolution=30):
    B = nlp.bounds
    x0 = nlp.getInitializationSample()

    dim = 2
    X = [None] * dim
    for i in range(dim):
        X[i] = np.linspace(B[0][i], B[1][i], resolution)
    X = np.stack(np.meshgrid(*X, indexing='ij'), axis=-1) 

    if nlp.dimension==2:
        fX = np.array([nlp.eval_scalar(x)[0] for x in X.reshape(-1, dim)])
    else:
        f = lambda x: nlp.eval_scalar(np.concatenate((x, x0[2:])))[0]
        fX = np.array([f(x) for x in X.reshape(-1, dim)])
    fX = fX.reshape(X.shape[:-1])

    fig, ax = plt.subplots()
    ax.contour(X[:,:,0], X[:,:,1], fX, 200)
    # ax.plot(x[:,0], x[:,1], 'o-r', ms=3)
    plt.show()

def display_any(nlp: op.NLP):
    ty = nlp.types

    komo = nlp.as_KOMO()
    if komo is not None:
        for _ in range(5):
            x = nlp.getUniformSample()
            # x = nlp.getInitializationSample()
            nlp.evaluate(x)
            komo.view(False, 'random init')
            time.sleep(.2)

    if (op.eq in ty) or (op.ineq in ty):
        nlp_org = nlp
        nlp = nlp.aug_lag(1e1, -1.)
    display_2d_unconstrained(nlp)

def main():
    problems = op.get_NLP_Problem_names()
    print(problems)

    # p = problems[5]
    for p in problems:
        nlp = op.make_NLP_Problem(p)
        print('===\n', p, nlp.report(1), '===')
        display_any(nlp)

if __name__ == "__main__":
    main()
