
TEST_MODULES = cellprofiler cellprofiler.modules cellprofiler.gui cellprofiler.matlab cellprofiler.cpmath

ifeq ($(OS),Windows_NT)
	PYTHON_COMPILER=mingw32
else
	PYTHON_COMPILER=unix
endif

all: exts tests

exts: cellprofiler/cpmath/_cpmorphology.pyd cellprofiler/cpmath/_watershed.pyd cellprofiler/cpmath/_propagate.pyd

cellprofiler/cpmath/_cpmorphology.pyd: cellprofiler/cpmath/src/cpmorphology.c
	python cellprofiler/cpmath/setup.py build_ext -i --compiler=$(PYTHON_COMPILER) 

cellprofiler/cpmath/_watershed.pyd: cellprofiler/cpmath/_watershed.pyx
	python cellprofiler/cpmath/setup.py build_ext -i --compiler=$(PYTHON_COMPILER) 

cellprofiler/cpmath/_propagate.pyd: cellprofiler/cpmath/_propagate.pyx
	python cellprofiler/cpmath/setup.py build_ext -i --compiler=$(PYTHON_COMPILER) 

tests:
	python -m nose.core --with-nosexunit --core-target=target/xml-report $(TEST_MODULES)
