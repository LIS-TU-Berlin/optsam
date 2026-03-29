from robotic import NLP, NLP_Solver, NLP_Sampler
from robotic import get_NLP_Problem_names, make_NLP_Problem
from robotic import OT, OptMethod
import sys

for t in dir(OT):
     if not t.startswith("_") and not t=='name' and not t=='value':
        setattr(sys.modules[__name__], t, getattr(OT,t))

