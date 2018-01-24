#                                        Ü       ÜÜÜÜÜÜÜÜÜÜÜ
#       Ü ÜÜÜÜÜÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÜ °    ÜÛ² Ü ÜÜÜÛÛÛßßßßßßÛÛÛÛÜ  °
#            ßßßßßßÛÛÛ² Ü   ßÛÛÛ² ±   ÛÛÛ²     ÛÛÛ² °     ÛÛÛ²  ±
#                  ÛÛÛ² °    ÛÛÛ² ß   ÛÛÛ²     ÛÛÛ² ß     ÛÛÛ²  ²
#           ÜÜÜÜÜÜÜÛÛÛ²ÜÜÜÜÜÜÛÛÛ² ÜÜÛÛÛÛÛÛÛÛÛÜÜßßÛ²ÜÛÛÛÛÛÛÛÛÛ²  ß
#          ÛÛÛßßßßßÛÛÛ²ßßßßßßÛÛÛ²ßßßßßÛÛÛ²ßßßßßßßÜ² Ü     ÛÛÛ²  Ü
#           ßÜ     ÛÛÛ² Ü    ÛÛÛ² Ü   ÛÛÛ² Ü   ÛÛÛ² ²     ÛÛÛ²  ²
#                  ÛÛÛ² ²    ÛÛÛ² ²   ÛÛÛ² ²   ÛÛÛ² ±     ÛÛÛ²  ±
#                  ÛÛÛ² ±    ÛÛÛ² ±   ÛÛÛ² ±   ÛÛÛ² °     ÛÛÛ²  °
#                  ÛÛÛ² °    ÛÛÛ² °   ÛÛÛ² °   ÛÛÛ²  Ü    ÛÛÛ²   Ü
#                 ÜÛÛÛÛÜÜß  ÜÛÛÛÛÜÜÜß  ÛÛ²  ÜÜÛÛÛÛÛÛß   ÜÛÛÛÛÛÛßß
#              Üßßßßßßßß  Üßßßßßßß      ßÛ ß           ß     [BROAD‘17]
#         Ü ÜÜÜ ÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜ ÜÜÜ Ü
#
#                  ... Broad Institute of MIT and Harvard ‘17
#
#                               Proudly Present ...
#
#        ÉÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍ»
#     ÚÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄ¿
#     ³                                                                  ³
#     ÀÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÙ
#        ÈÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍ¼
#                        Released at [þþ:þþ] on [þþ/þþ/þþ]
#     ÉÍÍÍÄÄÄÄÄÄÄÄÄÄÄÄÄÄ Ä  ÄÄ  Ä  Ä   ú   Ä  Ä  ÄÄ  Ä ÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÍÍÍ»
#     ³                                                                  º
#     ³   [Supplier    : 0x00B1........] [Operating System : all.....]   ³
#     ³   [Packager    : 0x00B1........] [Video            : none....]   ³
#     ³   [Cracker     : none..........] [Audio            : none....]   ³
#     ³   [Protection  : none..........] [Number of Disks  : 1.......]   ³
#     ú   [Type        : Dockerfile....] [Rating           : ........]   ú
#     ú                                                                  ú    
#     ú                                                                  ú
#     ú   Well, this is a little Dockerfile that have many functions     ú
#     ú   for quantifying phenotypes...enjoy....                         ú
#     ú                                                                  ú
#     ú                                                                  ú
#     ³                                                                  ³
#     ³                                                                  ³
#     ³                                                                  ³
#     ³                                                                  ³
#     º                                                                  º
#     ÈÍÍÍÄÄÄÄÄÄÄÄÄÄÄÄÄÄ Ä  ÄÄ  Ä  Ä   ú   Ä  Ä  ÄÄ  Ä ÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÍÍÍ¼
#
#     Greets: ...
#
#         mcquin
#         purohit
#
#                                 - [ BROAD‘17 ] -
#                                                          -0x00B1 [05/06/84]
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
ADD https://github.com/just-containers/s6-overlay/releases/download/v1.11.0.1/s6-overlay-amd64.tar.gz /tmp/

RUN gunzip -c /tmp/s6-overlay-amd64.tar.gz | tar -xf - -C /

ENTRYPOINT ["/init", "cellprofiler"]

CMD ["--run", "--run-headless", "--help"]
