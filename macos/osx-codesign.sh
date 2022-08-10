#!/usr/bin/env sh

cd ./dist/CellProfiler.app/Contents/Resources
sudo rm -r Home/legal/
sudo codesign --timestamp -f -s "Developer ID Application: Beth Cimini (27YQ9U45D9)" Home/lib/server/classes.jsa
find . -type f | xargs -I file codesign --timestamp -f -s "Developer ID Application: Beth Cimini (27YQ9U45D9)" file
cd ../MacOS
find . -type f | xargs -I file sudo codesign --timestamp -f -s "Developer ID Application: Beth Cimini (27YQ9U45D9)" file
codesign --timestamp -f -s "Developer ID Application: Beth Cimini (27YQ9U45D9)" _elementtree.cpython-38-darwin.so
codesign --entitlements entitlements.plist --timestamp -o runtime -f -s "Developer ID Application: Beth Cimini (27YQ9U45D9)" ./cp
cd ..
codesign --timestamp -f -s "Developer ID Application: Beth Cimini (27YQ9U45D9)" Info.plist


