[![Build Status](https://travis-ci.org/CellProfiler/docker.svg?branch=master)](https://travis-ci.org/CellProfiler/docker)

Testing docker locally:

    make
    make test

On a Linux host machine, running CellProfiler's GUI from the container:

    # Note, the following line is insecure.
    xhost +local:root
    docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro cellprofiler:latest ""
