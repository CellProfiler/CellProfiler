#!/usr/bin/env sh

cd ./dist/CellProfiler.app/Contents/Resources
sudo rm -r Home/legal/
sudo codesign --timestamp -s "Apple Development: alicelucas93@gmail.com (P6D4NCA4CT)" Home/lib/server/classes.jsa
find . -type f | xargs -I file codesign --timestamp -f -s "Apple Development: alicelucas93@gmail.com (P6D4NCA4CT)" file
cd ../MacOS
find . -type f | xargs -I file sudo codesign --timestamp -f -s "Apple Development: alicelucas93@gmail.com (P6D4NCA4CT)" file
codesign --timestamp -f -s "Apple Development: alicelucas93@gmail.com (P6D4NCA4CT)" _elementtree.cpython-38-darwin.so
codesign --entitlements entitlements.plist --timestamp -o runtime -s "Apple Development: alicelucas93@gmail.com (P6D4NCA4CT)" ./cp
cd ..
codesign --timestamp -s "Apple Development: alicelucas93@gmail.com (P6D4NCA4CT)" Info.plist


