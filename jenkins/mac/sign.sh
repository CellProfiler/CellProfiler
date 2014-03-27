#!/bin/bash
set -e
set -x

signdir="$1"
unsigned_zip="$2"
signed_zip="$3"

cd "$signdir"
unzip "$unsigned_zip"
rm "$unsigned_zip"
security unlock-keychain -p '' ~/Library/Keychains/cellprofiler.keychain

function sign()
{
    codesign --keychain ~/Library/Keychains/cellprofiler.keychain -f -s "BROAD INSTITUTE INC" CellProfiler.app
}
sign || sign

rm -f "$signed_zip"
zip -r "$signed_zip" CellProfiler.app
rm -rf CellProfiler.app
