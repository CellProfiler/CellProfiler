MAKEFLAGS += --warn-undefined-variables

SHELL := bash

.DEFAULT_GOAL := dist/CellProfiler.app

.DELETE_ON_ERROR:

.SHELLFLAGS := -eu -o pipefail -c

.SUFFIXES:

CellProfiler:
	git clone https://github.com/emilroz/CellProfiler

	cd $@ && git checkout compat-4-0

.PHONY: dependencies
dependencies: CellProfiler

	git clone https://github.com/glencoesoftware/omero-reader.git
	cd omero-reader && git checkout master && pip install --editable . --upgrade

	git clone https://github.com/emilroz/python-bioformats.git
	cd python-bioformats && git checkout compat-4-0 && pip install --editable . --upgrade

	git clone https://github.com/emilroz/core.git

	cd core && git checkout compat-4-0
	pip3 install --editable "core[test]" --upgrade

	pip3 install --editable "CellProfiler[build, test]" --upgrade

dist/CellProfiler.app: CellProfiler dependencies Info.plist
	pyinstaller --noconfirm CellProfiler.spec

	cp $(word 3, $^) $@/Contents

	cp entitlements.plist $@/Contents/MacOS

.PHONY: clean
clean:
	if [ -d CellProfiler ]; then rm -rf CellProfiler; fi

	if [ -d build ]; then rm -rf build; fi
	
	if [ -d dist ]; then rm -rf build; fi

	if [ CellProfiler.dmg ]; then rm -rf CellProfiler.dmg; fi

	if [ -d core ]; then rm -rf core; fi
