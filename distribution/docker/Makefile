# Inputs from https://github.com/CellProfiler/examples/tree/master/ExampleHuman
EXAMPLE_CDN = https://github.com/CellProfiler/examples/archive/master.zip

# Public gold output files
S3_GOLD = https://s3.amazonaws.com/cellprofiler-examples/example-human-gold-standard

CP_VERSION ?= 4.2.3

TAG := latest

.DEFAULT_GOAL: build
build:
	docker build -t cellprofiler:$(CP_VERSION) --build-arg cp_version=$(CP_VERSION) .
	#docker tag cellprofiler:$(CP_VERSION) cellprofiler:$(CP_VERSION)

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
		cellprofiler:$(CP_VERSION) \
		--image-directory=/input \
		--output-directory=/output \
		--pipeline=/input/ExampleHuman.cppipe \
		--file-list=/input/filelist.txt
	
	# Compare gold files against output that was run.
	python -c "import imageio.v3 as iio; import numpy as np; im1 = iio.imread('output/AS_09125_050116030001_D03f00d0_Overlay.png'); im2 = iio.imread('output/gold/AS_09125_050116030001_D03f00d0_Overlay.png'); res = np.array_equal(im1, im2) and exit() ; exit(1)"
