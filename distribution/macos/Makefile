# This is a Makefile for building CellProfiler.app bundle on MacOS
# it is used by the CI process, but can also be used locally
# THE CP_VERSION CP_BRANCH CP_DIR variables may be overridden on the command line
#
# CP_DIR:      If not provided - the parent of this directory, `../..` is used
#              If provided - The path to the directory containing the CellProfiler source code
#                            which may be the parent of this directory, i.e. `../..`
#                            or may be a path to a seperate copy of the CellProfiler source code.
# CP_VERSION:  Should match the version of CellProfiler in the CP_DIR, i.e. as reported by _version.py, setuptools-scm, etc.
#              It is used to set the CFBundleVersion in Info.plist.template, which produces the final Info.plist file
# CP_BRANCH:   If not provided - then the CP_DIR's working directory will be set to the default branch, "main"
#              If provided with name - then the CP_DIR's working directory will be set to this branch
#              If provided with empty string - then the CP_DIR's working directory will not be modified
#
# Example Usage:
# make CP_VERSION=$(python -m setuptools_scm -c ../../src/frontend/pyproject.toml) CP_BRANCH=""


MAKEFLAGS += --warn-undefined-variables

SHELL := bash

CP_VERSION ?= 4.2.3

CP_BRANCH ?= main

CP_DIR    ?= ../..

CP_DIST   ?= ${CP_DIR}/distribution/macos

.DEFAULT_GOAL := dist/CellProfiler.app

.DELETE_ON_ERROR:

.SHELLFLAGS := -eu -o pipefail -c

.SUFFIXES:

CellProfiler:
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
	if [ -d build ]; then rm -rf build; fi

	if [ -d dist ]; then rm -rf dist; fi

	if [ CellProfiler.dmg ]; then rm -rf CellProfiler.dmg; fi

	if [ -f Info.plist ]; then rm Info.plist; fi
