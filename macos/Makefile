MAKEFLAGS += --warn-undefined-variables

SHELL := bash

.DEFAULT_GOAL := CellProfiler.dmg

.DELETE_ON_ERROR:

.SHELLFLAGS := -eu -o pipefail -c

.SUFFIXES:

identity = "Apple Distribution: Allen Goodman (MERWFR65Q3)"

CellProfiler:
	git clone git@github.com:CellProfiler/CellProfiler.git $@

	cd $@ && git checkout master

.PHONY: dependencies
dependencies: CellProfiler
	pip3 install --editable "CellProfiler[build, test]" --upgrade

dist/CellProfiler.app: CellProfiler dependencies Info.plist
	pyinstaller --noconfirm CellProfiler.spec

	cp $(word 3, $^) $@/Contents

	# codesign --deep -s $(identity) $@

	# xattr -lr $@
	# xattr -cr $@

.PHONY: sign
sign:
    codesign --force --verify --verbose --deep -s "Apple Distribution: Allen Goodman (MERWFR65Q3)" dist/CellProfiler.app

CellProfiler.dmg: dist/CellProfiler.app
	hdiutil create                  		\
		-format UDRW                		\
		-fs HFS+                    		\
		-fsargs "-c c=64,a=16,e=16" 		\
		-size 400000k               		\
		-srcfolder dist/CellProfiler.app 	\
		-volname CellProfiler      			\
		CellProfiler.dmg

	codesign --force --verify --verbose --deep -s $(identity) $@

.PHONY: clean
clean:
	if [ -d build ]; then rm -rf build; fi
	
	if [ -d dist ]; then rm -rf build; fi

	if [ CellProfiler.dmg ]; then rm -rf CellProfiler.dmg; fi
