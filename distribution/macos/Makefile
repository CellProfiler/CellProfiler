MAKEFLAGS += --warn-undefined-variables

SHELL := bash

CP_VERSION ?= 4.2.3

CP_BRANCH ?= main

CP_DIR    ?= CellProfiler

CP_DIST   ?= ${CP_DIR}/distribution/macos

.DEFAULT_GOAL := dist/CellProfiler.app

.DELETE_ON_ERROR:

.SHELLFLAGS := -eu -o pipefail -c

.SUFFIXES:

CellProfiler:
	if ! [ -d ${CP_DIR} ]; then git clone https://github.com/CellProfiler/CellProfiler.git $(CP_DIR); fi

	if [ ! -z "$(CP_BRANCH)" ]; then cd ${CP_DIR} && git checkout ${CP_BRANCH}; fi

.PHONY: dependencies
dependencies: CellProfiler
	pip3 install --editable "${CP_DIR}/src/subpackages/library" --upgrade
	pip3 install --editable "${CP_DIR}/src/subpackages/core" --upgrade
	pip3 install --editable "${CP_DIR}/src/frontend[build]" --upgrade

Info.plist: Info.plist.template
	sed 's/{{CP_VERSION}}/${CP_VERSION}/g' ${CP_DIST}/Info.plist.template > ${CP_DIST}/Info.plist

dist/CellProfiler.app: CellProfiler dependencies Info.plist
	cd ${CP_DIST}

	pyinstaller --noconfirm CellProfiler.spec

	cp Info.plist $@/Contents

	cp entitlements.plist $@/Contents/MacOS

.PHONY: clean
clean:
	# if [ -d CellProfiler ]; then rm -rf CellProfiler; fi

	if [ -d build ]; then rm -rf build; fi

	if [ -d dist ]; then rm -rf dist; fi

	if [ CellProfiler.dmg ]; then rm -rf CellProfiler.dmg; fi

	if [ -f Info.plist ]; then rm Info.plist; fi
