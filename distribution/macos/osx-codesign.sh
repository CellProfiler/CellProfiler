#!/usr/bin/env sh
set -eu

codesign --verify --deep --strict --verbose=2 "${1:-./dist/CellProfiler.app}"
