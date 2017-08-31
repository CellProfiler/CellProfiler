MAKEFLAGS += --warn-undefined-variables

SHELL := bash

.DEFAULT_GOAL := CellProfiler.dmg

.DELETE_ON_ERROR:

.SHELLFLAGS := -eu -o pipefail -c

.SUFFIXES:

identity = "Developer ID Application: Allen Goodman (MERWFR65Q3)"

CellProfiler:
	git clone git@github.com:CellProfiler/CellProfiler.git $@

.PHONY: dependencies
dependencies: CellProfiler
	pip2 install --upgrade cython
	pip2 install --upgrade numpy
	pip2 install --upgrade scipy

	pip2 install --editable "CellProfiler[build, test]" --upgrade   

dist/CellProfiler.app: CellProfiler dependencies Info.plist
	pyinstaller --noconfirm CellProfiler.spec

	cp $(word 3, $^) $@/Contents

	codesign --deep -s $(identity) $@

CellProfiler.dmg: dist/CellProfiler.app
	hdiutil create                  \
		-format UDRW                \
		-fs HFS+                    \
		-fsargs "-c c=64,a=16,e=16" \
		-size 170000k               \
		-srcfolder $<               \
		-volname CellProfiler       \
		$@

.PHONY: clean
clean:
	rm -rf build dist
