FROM ubuntu:16.04

# Install CellProfiler dependencies
RUN   apt-get -y update &&                                          \
      apt-get -y install                                            \
        build-essential    \
        cython             \
        git                \
        libmysqlclient-dev \
        libhdf5-dev        \
        libxml2-dev        \
        libxslt1-dev       \
        openjdk-8-jdk      \
        python-dev         \
        python-pip         \
        python-h5py        \
        python-matplotlib  \
        python-mysqldb     \
        python-scipy       \
        python-numpy       \
        python-wxgtk3.0    \
        python-zmq

WORKDIR /usr/local/src

# Install CellProfiler
RUN git clone https://github.com/CellProfiler/CellProfiler.git

WORKDIR /usr/local/src/CellProfiler

ARG version=3.0.0

RUN git checkout tags/v$version

RUN pip install --editable .

# Fix init and zombie process reaping problems using s6 overlay
ADD https://github.com/just-containers/s6-overlay/releases/download/v1.21.2.2/s6-overlay-amd64.tar.gz /tmp/

RUN tar -xzf /tmp/s6-overlay-amd64.tar.gz -C /

ENTRYPOINT ["/init", "cellprofiler"]

CMD ["--run", "--run-headless", "--help"]
