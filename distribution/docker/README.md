Authors: [@purohit](https://github.com/purohit), [@mcquin](https://github.com/mcquin), [@0x00b1](https://github.com/0x00b1), [@bethac07](https://github.com/bethac07)

This is configured to build a docker image for a specific version of CellProfiler.

The Makefile builds the Docker image, and the version can be specified by setting a `CP_VERSION` environment variable. For example:

```sh
export CP_VERSION="4.2.6"
```

Alternatively, you can edit these lines in the two files directly:

Makefile:

    CP_VERSION := x.y.z

Dockerfile:

    ARG cp_version=x.y.z

You can do this using `sed`, e.g. if you would like to specify version `4.2.6`:

```sh
# assumes you have done `export CP_VERSION="x.y.z"`
sed -i'' -e 's/^ARG cp_version=.*/ARG cp_version=${CP_VERSION}/g' Dockerfile
sed -i'' -e 's/^CP_VERSION \?=.*/CP_VERSION ?= ${CP_VERSION}/g' Makefile
```

NOTE: The commands above work on Linux, using the default GNU sed. MacOS uses BSD sed by default which parses the `-i` flag differently. Simply place a space after the `-i` flag to work around this, i.e. `-i `.

To build and test docker locally:

```sh
# assumes you have done `export CP_VERSION="x.y.z"`
# or have replaced the variable CP_VERSION to desired version
make
make clean
make test
```

Note: `make test` will work only for CellProfiler 4.0.0 and higher because the example pipeline in the test is not backward compatible.

By default, s6 logging is output to stdout. To change this behavior, [set the environment variable](https://github.com/just-containers/s6-overlay#customizing-s6-behaviour) `S6_LOGGING` e.g.:

```sh
# assumes you have done `export CP_VERSION="x.y.z"`
docker run -e S6_LOGGING=1 cellprofiler:${CP_VERSION}
```


To push to docker hub, do the following (look up instructions [here](https://docs.docker.com/docker-cloud/builds/push-images/) for details)

```sh
# assumes you have done `export CP_VERSION="x.y.z"`
export DOCKER_ID_USER="username" # replace with your Docker Hub username
docker login
docker tag cellprofiler:${CP_VERSION}  ${DOCKER_ID_USER}/cellprofiler:${CP_VERSION}
docker push ${DOCKER_ID_USER}/cellprofiler:${CP_VERSION}
```

On a Linux host machine, running CellProfiler's GUI from the container:

```sh
# assumes you have done `export CP_VERSION="x.y.z"`
# Note, the following line is insecure.
xhost +local:root
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro cellprofiler:${CP_VERSION} ""
```
