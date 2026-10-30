MAKEFLAGS += --warn-undefined-variables
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := app
.DELETE_ON_ERROR:

CP_DIR ?= $(abspath ../..)
CP_DIST := $(CP_DIR)/distribution/macos
CP_BUILD ?= $(CP_DIST)/build
CP_OUTPUT ?= $(CP_DIST)/dist
PIXI ?= pixi
PYINSTALLER_FLAGS ?= --noconfirm
PIXI_RUN = $(PIXI) run --locked --manifest-path "$(CP_DIR)/pixi.toml" -e build

.PHONY: app java clean

java:
	$(PIXI_RUN) python "$(CP_DIST)/prepare_java.py" "$(CP_BUILD)/java-jars"

app: java
	CP_JAVA_JARS="$(CP_BUILD)/java-jars" $(PIXI_RUN) python -m PyInstaller $(PYINSTALLER_FLAGS) --workpath "$(CP_BUILD)/pyinstaller" --distpath "$(CP_OUTPUT)" "$(CP_DIST)/CellProfiler.spec"

clean:
	rm -rf "$(CP_BUILD)" "$(CP_OUTPUT)"
