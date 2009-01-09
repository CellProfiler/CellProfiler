
TEST_MODULES = cellprofiler cellprofiler.modules cellprofiler.gui cellprofiler.matlab cellprofiler.cpmath cellprofiler.gui

all: tests

tests:
	python -m nose.core --with-nosexunit --core-target=target/xml-report $(TEST_MODULES)
