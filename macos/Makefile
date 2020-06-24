MAKEFLAGS += --warn-undefined-variables

SHELL := bash

.DEFAULT_GOAL := dist/CellProfiler.app

.DELETE_ON_ERROR:

.SHELLFLAGS := -eu -o pipefail -c

.SUFFIXES:

CellProfiler:
	git clone https://github.com/CellProfiler/CellProfiler.git

	cd $@ && git checkout master

.PHONY: dependencies
dependencies: CellProfiler
	pip3 install --editable "CellProfiler[build, test]" --upgrade

dist/CellProfiler.app: CellProfiler dependencies Info.plist
	pyinstaller --noconfirm CellProfiler.spec

	cp $(word 3, $^) $@/Contents

.PHONY: clean
clean:
	if [ -d build ]; then rm -rf build; fi
	
	if [ -d dist ]; then rm -rf build; fi

	if [ CellProfiler.dmg ]; then rm -rf CellProfiler.dmg; fi
