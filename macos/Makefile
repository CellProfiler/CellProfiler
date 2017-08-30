identity = "Developer ID Application: Allen Goodman (MERWFR65Q3)"

build:
	git clone git@github.com:CellProfiler/CellProfiler.git $@

.INTERMEDIATE: CellProfiler.app
CellProfiler.app: build Info.plist
	pyinstaller --distpath . --noconfirm CellProfiler.spec

	rm CellProfiler

	cp $(word 2, $^) $@/Contents

	codesign --deep -s $(identity) $@

CellProfiler.dmg: CellProfiler.app
	hdiutil create                  \
		-format UDRW                \
		-fs HFS+                    \
		-fsargs "-c c=64,a=16,e=16" \
		-size 170000k               \
		-srcfolder $<               \
		-volname CellProfiler       \
		$@

.PHONY: clean
clean:
	rm -rf build

.PHONY: dependencies
dependencies: build
	pip2 install --editable "build[build, test]" --upgrade   
