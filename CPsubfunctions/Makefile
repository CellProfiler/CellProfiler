
MEXEXT := $(shell mexext)

all: CPljosaprobseg.$(MEXEXT) CPljosaprobseg.m

CPljosaprobseg.$(MEXEXT): ljosaprobseg-1.0/ljosaprobseg.$(MEXEXT)
	cp $< $@

ljosaprobseg-1.0/ljosaprobseg.$(MEXEXT):
	make -C ljosaprobseg-1.0

CPljosaprobseg.m: ljosaprobseg-1.0/ljosaprobseg.m
	cp $< $@
