#!/bin/sh
set -e

build_shell=""
if [ ! -z "$1" ]
then
	echo "Interactive shell at step: $1"
	build_shell="--build-shell=$1"
fi
set -x

flatpak-builder -y \
	--install-deps-from=flathub \
	--force-clean \
	--repo=tmp_repo \
	$build_shell \
	build-dir \
	org.cellprofiler.cellprofiler.yaml

rm -rf CellProfiler.flatpak
flatpak build-bundle \
	--runtime-repo=https://flathub.org/repo/flathub.flatpakrepo \
	tmp_repo/ \
	CellProfiler.flatpak \
	org.cellprofiler.cellprofiler

rm -r tmp_repo/
