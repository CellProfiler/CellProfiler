#!/usr/bin/env sh

cd ./dist/CellProfiler.app/Contents/Resources || exit 1
sudo rm -r Home/legal/
sudo codesign --timestamp -f -s "Developer ID Application: Beth Cimini (27YQ9U45D9)" Home/lib/server/classes.jsa
find . -type f -print0 | xargs -0 -I file codesign --timestamp -f -s "Developer ID Application: Beth Cimini (27YQ9U45D9)" file
cd ../MacOS || exit 1
find . -type f -print0 | xargs -0 -I file sudo codesign --timestamp -f -s "Developer ID Application: Beth Cimini (27YQ9U45D9)" file
codesign --timestamp -f -s "Developer ID Application: Beth Cimini (27YQ9U45D9)" _elementtree.cpython-39-darwin.so
codesign --entitlements entitlements.plist --timestamp -o runtime -f -s "Developer ID Application: Beth Cimini (27YQ9U45D9)" ./cp
cd ..
codesign --timestamp -f -s "Developer ID Application: Beth Cimini (27YQ9U45D9)" Info.plist
