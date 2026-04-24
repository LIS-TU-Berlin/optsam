ipynb_paths =  $(shell find . -maxdepth 1 -type f -name '*.ipynb' -not -name '*checkpoint*' -not -name '*nbconvert*' -printf '%f ')

py_paths =  $(shell find . -maxdepth 1 -type f -name '*.py' -not -name '*checkpoint*' -not -name '*nbconvert*' -printf '%f ')

sync: $(py_paths:%=%.sync)
to_nb: $(py_paths:%=%.to_nb)
pair: $(ipynb_paths:%=%.pair)
nometa: $(ipynb_paths:%=%.nometa)

%.py.sync: %.py
	+@-jupytext --sync $*.py

%.ipynb.pair: %.py
	+@-jupytext --set-formats ipynb,py:percent $*.ipynb

%.py.to_nb: %.py
	+@-jupytext --to ipynb $*.py
