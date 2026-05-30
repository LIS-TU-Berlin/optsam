import optsam as op
import numpy as np
import time

def generate_samples(nlp : op.NLP, n):
    solver = op.NLP_Sampler(nlp)
    #solver.setOptions() #access to many options - let's take default
    data = []
    while len(data)<n:
        ret = solver.sample()
        print(ret)
        if ret.feasible:
            data.append(ret.x)
    return np.stack(data)

def display_samples(nlp : op.NLP, data):
    komo = nlp.as_KOMO()
    assert komo is not None, 'this works only for komo problems'
    for x in data:
        phi, J = nlp.evaluate(x)
        err_eq = np.sum(phi[nlp.types==op.OT.eq])
        print('data:', x, '\nphi:', phi, '\nerr:', err_eq)
        komo.view(False, f'data {x}')
        time.sleep(1.)

def main():
    problems = op.get_NLP_Problem_names()
    print('== these are all problems:', problems)

    nlp = op.make_NLP_Problem('IK')
    print('== we pick "IK" as a problem:\n', nlp.report(1), '===')

    data = generate_samples(nlp, 10)
    display_samples(nlp, data)

if __name__ == "__main__":
    main()