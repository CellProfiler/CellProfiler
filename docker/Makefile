# Inputs from https://github.com/CellProfiler/examples/tree/master/ExampleHuman
EXAMPLE_CDN = https://github.com/CellProfiler/examples/archive/master.zip

# Public gold output files
S3_GOLD = https://s3.amazonaws.com/cellprofiler-examples/example-human-gold-standard

VERSION := 4.0.0

TAG := latest

.DEFAULT_GOAL: build
build:
	docker build -t cellprofiler:$(VERSION) --build-arg version=$(VERSION) .
	docker tag cellprofiler:$(VERSION) cellprofiler:$(VERSION)

.PHONY: input
input:
	mkdir -p $@

output:
	mkdir -m 777 -p $@

# The files to compare against after a run of CellProfiler
# Note that while Image.csv is also output, it is not compared against,
# because it contains hashes that change per-run.
output/gold: output
	mkdir $@

AS_09125_050116030001_D03f00d0_Overlay.png:	output/gold
	curl -o $</$@ ${S3_GOLD}/$@

master.zip:
	curl -LOk ${EXAMPLE_CDN}

data: master.zip
	unzip $< -d input
	mv input/examples-master/ExampleHuman/images/* input/
	mv input/examples-master/ExampleHuman/ExampleHuman.cppipe input/

input/filelist.txt: data
	echo 'file:///input/AS_09125_050116030001_D03f00d0.tif' >> $@
	echo 'file:///input/AS_09125_050116030001_D03f00d1.tif' >> $@
	echo 'file:///input/AS_09125_050116030001_D03f00d2.tif' >> $@

.PHONY: clean
clean:
	rm -rf input
	rm -rf output
	rm -f master.zip

.PHONY: test
test: input output output/gold data input/filelist.txt AS_09125_050116030001_D03f00d0_Overlay.png
	docker run \
		--volume=`pwd`/input:/input \
		--volume=`pwd`/output:/output \
		cellprofiler:$(VERSION) \
		--image-directory=/input \
		--output-directory=/output \
		--pipeline=/input/ExampleHuman.cppipe \
		--file-list=/input/filelist.txt
	
	# Compare gold files against output that was run.
	diff -b output/AS_09125_050116030001_D03f00d0_Overlay.png output/gold/AS_09125_050116030001_D03f00d0_Overlay.png
