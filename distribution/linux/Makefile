# This is a Makefile for building a flatpak for CellProfiler
# it is used by the CI process, but can also be used locally (assuming flatpak and flatpak-builder are available)
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
# REPO_DIR:    If provided, the location of the local flatpak repo to install into
#
# Example Usage:
# make CP_VERSION=$(python -m setuptools_scm -c ../../src/frontend/pyproject.toml) CP_BRANCH=""


MAKEFLAGS += --warn-undefined-variables

SHELL       := bash
.SHELLFLAGS := -eu -o pipefail -c

.DELETE_ON_ERROR:
.SUFFIXES:

.DEFAULT_GOAL := build-local

CP_VERSION ?= 5.0.0
CP_BRANCH  ?= main
CP_DIR     ?= ../..
REPO_DIR   ?= $(CP_DIR)/../cprepo

CP_DIST    := $(CP_DIR)/distribution/linux

RESOLVED_CP_BRANCH := $(shell \
	if [ -z "$(CP_BRANCH)" ]; then \
		git rev-parse --abbrev-ref HEAD; \
	else \
		echo "$(CP_BRANCH)"; \
	fi)

CP_VERSION_BASE := $(shell \
  if [[ "$(CP_VERSION)" =~ ^(.+)\.dev[0-9]+$$ ]]; then \
    echo "$${BASH_REMATCH[1]}"; \
  else \
    echo "$(CP_VERSION)"; \
  fi)

CP_VERSION_DEV_TAG := $(shell \
  if [[ "$(CP_VERSION)" =~ ^.+\.(dev[0-9]+)$$ ]]; then \
    echo "~$${BASH_REMATCH[1]}"; \
  fi)

CP_VERSION_TYPE := $(shell \
  if [[ "$(CP_VERSION)" =~ \.dev[0-9]+$$ ]]; then \
    echo "development"; \
  else \
    echo "stable"; \
  fi)

MANIFEST_DATE := $(shell date +%F)


CellProfiler:
	if [ ! -z "$(CP_BRANCH)" ]; then cd $(CP_DIR) && git checkout $(RESOLVED_CP_BRANCH); fi

#.PHONY: dependencies
#dependencies: CellProfiler
#	pip3 install --editable "$(CP_DIR)/src/subpackages/library" --upgrade
#	pip3 install --editable "$(CP_DIR)/src/subpackages/core" --upgrade
#	pip3 install --editable "$(CP_DIR)/src/frontend[build]" --upgrade

.PHONY: metadata
metadata: org.cellprofiler.CellProfiler.metainfo.xml org.cellprofiler.CellProfiler.yml

org.cellprofiler.CellProfiler.metainfo.xml: org.cellprofiler.CellProfiler.metainfo.xml.template
	sed -e 's/{{METAINFO_VERSION}}/$(CP_VERSION_BASE)$(CP_VERSION_DEV_TAG)/g' \
	    -e 's/{{METAINFO_TYPE}}/$(CP_VERSION_TYPE)/g' \
	    -e 's/{{METAINFO_DATE}}/$(MANIFEST_DATE)/g' \
	    $< > $(CP_DIST)/org.cellprofiler.CellProfiler.metainfo.xml

org.cellprofiler.CellProfiler.yml: org.cellprofiler.CellProfiler.yml.template
	sed -e 's/{{CP_VERSION}}/$(CP_VERSION)/g' \
	    -e 's/{{CP_VERSION_BASE}}/$(CP_VERSION_BASE)/g' \
	    -e 's/{{CP_BRANCH}}/$(RESOLVED_CP_BRANCH)/g' \
	    $< > $(CP_DIST)/org.cellprofiler.CellProfiler.yml

.PHONY: build-local
build-local: CellProfiler metadata
	cd $(CP_DIST)
	flatpak-builder --force-clean --install-deps-from=flathub --repo=${REPO_DIR} --user --install builddir org.cellprofiler.CellProfiler.yml

.PHONY: clean
clean:
	if [ -f org.cellprofiler.CellProfiler.metainfo.xml ]; then rm org.cellprofiler.CellProfiler.metainfo.xml; fi
	if [ -f org.cellprofiler.CellProfiler.yml ]; then rm org.cellprofiler.CellProfiler.yml; fi

.PHONY: clean-build
clean-build:
	if [ -d .flatpak-builder ]; then rm -rf .flatpak-builder; fi
	if [ -d builddir ]; then rm -rf builddir; fi
	if [ -d $(REPO_DIR) ]; then rm -rf $(REPO_DIR); fi
