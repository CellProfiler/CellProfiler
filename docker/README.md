## CellProfiler Dockerfile

### Build

        docker build -t cellprofiler .

### Usage

#### Get help

        docker run --rm cellprofiler

#### Run with an example

        cd /tmp
        wget http://cellprofiler.org/linked_files/Examplezips/ExampleHumanImages.zip
        unzip ExampleHumanImages.zip
        docker run -v /tmp/ExampleHumanImages:/tmp/ExampleHumanImages cellprofiler -p /tmp/ExampleHumanImages/ExampleHuman.cppipe
