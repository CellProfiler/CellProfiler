Authors: [@purohit](https://github.com/purohit), [@mcquin](https://github.com/mcquin), [@0x00b1](https://github.com/0x00b1)

This is configured to build a docker image for `3.0.0`. To build a different version, edit these lines:

Makefile:

    VERSION := 3.0.0

Dockerfile:

    ARG version=3.0.0
    
You can do this using `sed` e.g.
    
    export VERSION=2.2.1
    sed -i s,"ARG version=3.0.0","ARG version=${VERSION}",g Dockerfile
    sed -i s,"VERSION := 3.0.0","VERSION := ${VERSION}",g Makefile

Note: the sed commands above are for GNU sed. If you're on OS X, see [this](https://stackoverflow.com/questions/30003570/how-to-use-gnu-sed-on-mac-os-x).

To build and test docker locally:

    make
    make clean
    make test

Note: `make test` will work only for CellProfiler 3.3.0 and higher because the example pipeline in the test is not backward compatible.

By default, s6 logging is output to stdout. To change this behavior, [set the environment variable](https://github.com/just-containers/s6-overlay#customizing-s6-behaviour) `S6_LOGGING` e.g.:

    docker run -e S6_LOGGING=1 cellprofiler:${VERSION}


To push to docker hub, do the following (look up instructions [here](https://docs.docker.com/docker-cloud/builds/push-images/) for details)

    export DOCKER_ID_USER="username" # replace with your Docker Hub username 
    export VERSION=3.0.0 # replace with the version you are building
    docker login
    docker tag cellprofiler:${VERSION}  ${DOCKER_ID_USER}/cellprofiler:${VERSION} 
    docker push ${DOCKER_ID_USER}/cellprofiler:${VERSION} 
    
 On a Linux host machine, running CellProfiler's GUI from the container:

    # Note, the following line is insecure.
    xhost +local:root
    VERSION=3.0.0 docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro cellprofiler:${VERSION} ""
