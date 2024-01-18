MAKEFLAGS += --warn-undefined-variables

SHELL := bash

CP_VERSION ?= 4.2.3

.DEFAULT_GOAL := dist/CellProfiler.app

.DELETE_ON_ERROR:

.SHELLFLAGS := -eu -o pipefail -c

.SUFFIXES:

CellProfiler:
	git clone https://github.com/CellProfiler/CellProfiler.git

	cd $@ && git checkout main

.PHONY: dependencies
dependencies: CellProfiler
	pip3 install --editable "CellProfiler/src/frontend[build]" --upgrade

Info.plist: Info.plist.template
	sed 's/{{CP_VERSION}}/${CP_VERSION}/g' Info.plist.template > Info.plist

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

	if [ -f Info.plist ]; then rm Info.plist; fi
