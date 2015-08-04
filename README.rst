CellProfiler is free, open-source software designed to enable biologists without training in computer vision or programming to quantitatively measure phenotypes from thousands of images automatically.

Our website, [cellprofiler.org](http://cellprofiler.org/), is aimed at non-developer users and has lots of information about CellProfiler and how to use it, as well as a [forum](http://cellprofiler.org/forum/) for asking questions.

The [developer wiki](https://github.com/CellProfiler/CellProfiler/wiki) covers topic such as how to build CellProfiler from source, how to set up a development environment with all of CellProfiler's dependencies, and the various resources set up to support CellProfiler development.

## Getting started with the Docker image of CellProfiler

The Docker image is for advanced users who want to run CellProfiler for high-throughput processing without a graphical user interface in a Linux cluster environment. This is an alternative to installing CellProfiler from source or using [the Linux build that we provide](http://cellprofiler.org/linux.shtml). New users will typically want to download and install CellProfiler on a Mac or Windows machine instead; see [the download page](http://cellprofiler.org/download.shtml).

To use the Docker image, you need to Docker's `-v` option to mount the input directory (which contains your images, pipeline, and other input files) and the output directory (to which images, measurements, and other output files will be written). Because CellProfiler will be run as the user "cellprofiler", you must ensure that the "cellprofiler" user has permission to write to the output directory.

Example for how to use the Docker image:

```bash
mkdir input
cd input
curl -O http://cellprofiler.org/svnmirror/ExampleImages/ExampleHumanImages/ExampleHuman.cppipe
curl -O http://cellprofiler.org/svnmirror/ExampleImages/ExampleHumanImages/AS_09125_050116030001_D03f00d0.tif
curl -O http://cellprofiler.org/svnmirror/ExampleImages/ExampleHumanImages/AS_09125_050116030001_D03f00d1.tif
curl -O http://cellprofiler.org/svnmirror/ExampleImages/ExampleHumanImages/AS_09125_050116030001_D03f00d2.tif
for a in *.tif; do echo file:///input/$a; done > filelist.txt
cd ..
mkdir -m 777 output
docker run --rm -v $(pwd)/input:/input -v $(pwd)/output:/output ljosa/cellprofiler:master -i /input -o /output -p /input/ExampleHuman.cppipe --file-list=/input/filelist.txt
```

## How to file new issues

If you have a bug or other issue to report, please read the wiki page on [how to file new issues](https://github.com/CellProfiler/CellProfiler/wiki/How-to-file-new-issues) to learn the best way to report it. You can also search the [forum](http://cellprofiler.org/forum/) which may have a report or work-around for the issue.
