PY_VER = $(shell /usr/bin/env python3 -c "import sys; print(str(sys.version_info[0])+'.'+str(sys.version_info[1]))")
PY_SITE = $(VIRTUAL_ENV)/lib/python$(PY_VER)/site-packages

.NOTPARALLEL:

info:
	@echo "PY_VER:" $(PY_VER)
	@echo "PY_SITE:" $(PY_SITE)

docs: force
	sphinx-build docs html

docs-clean:
	rm -Rf html

local-install:
	uv pip install -e .

local-clean:
	-rm -Rf $(PY_SITE)/optsam
	-rm -Rf $(PY_SITE)/optsam-*
	-rm -Rf robotic/__pycache__ build/bdist* build/lib robotic.egg-info

wheel:
	python -m build

wheel-upload:
	twine upload dist/*.whl

force:	;
