FROM centos:centos6

ADD docker/cellprofiler.repo /etc/yum.repos.d/
RUN yum install -y cellprofiler-cython-0.20.2-1 cellprofiler-python-2.7.2-1 cellprofiler-dateutil-2.2-1 cellprofiler-decorator-3.2.0-1 cellprofiler-h5py-2.2.0-2 cellprofiler-hdf5-1.8.10patch1-1 cellprofiler-ilastik-0.5.05-4 cellprofiler-jdk-7u21-1 cellprofiler-libjpeg-8b-1 cellprofiler-libpng-1.4.5-1 cellprofiler-libtiff-3.9.4-1 cellprofiler-matplotlib-1.0.1-2 cellprofiler-mysqlpython-1.2.3-1 cellprofiler-numpy-1.9.0-1 cellprofiler-pil-1.1.7-2 cellprofiler-pyopengl-3.0.1-3 cellprofiler-pyqt-x11-gpl-4.8.3-2 cellprofiler-pysqlite-2.6.1-1 cellprofiler-pytz-2013.7-1 cellprofiler-pyzmq-2.1.11-1 cellprofiler-qimage2ndarray-1.0-2 cellprofiler-scikit-learn-0.15.2-2 cellprofiler-scipy-0.13.2-1 cellprofiler-setuptools-1.1.6-1 cellprofiler-vigra-1.7.1-2 cellprofiler-wxpython-2.8.11.0-1 xorg-x11-fonts-Type1 liberation-fonts-common liberation-sans-fonts gcc cellprofiler-hdf5-devel-1.8.10patch1-1 cellprofiler-numpy-devel-1.9.0-1 patch gcc-c++ git cellprofiler-javabridge-1.0.9 cellprofiler-bioformats-1.0.3 xorg-x11-server-Xvfb

RUN mkdir -p /usr/cellprofiler/src
ADD . /usr/cellprofiler/src/CellProfiler

# RUN cd /usr/cellprofiler/src/CellProfiler; patch -p0 < docker/threshold.diff

RUN cd /usr/cellprofiler/src/CellProfiler; PATH=/usr/cellprofiler/bin:/usr/cellprofiler/jdk/bin:$PATH LD_LIBRARY_PATH=/usr/cellprofiler/jdk/lib:/usr/cellprofiler/jdk/jre/lib/amd64/server: JAVA_HOME=/usr/cellprofiler/jdk /usr/cellprofiler/bin/python CellProfiler.py --build-and-exit

RUN cd /usr/cellprofiler/src/CellProfiler; PATH=/usr/cellprofiler/bin:/usr/cellprofiler/jdk/bin:$PATH LD_LIBRARY_PATH=/usr/cellprofiler/jdk/lib:/usr/cellprofiler/jdk/jre/lib/amd64/server: JAVA_HOME=/usr/cellprofiler/jdk MAVEN_OPTS="-Xmx1024m" /usr/cellprofiler/bin/python external_dependencies.py -o

RUN cd /usr/cellprofiler/src/CellProfiler; patch -p0 < docker/cellprofiler-frozen.diff

RUN (cd /usr/cellprofiler/src/CellProfiler/cellprofiler/utilities && PATH=/usr/cellprofiler/bin:/usr/cellprofiler/jdk/bin:$PATH LD_LIBRARY_PATH=/usr/cellprofiler/jdk/lib:/usr/cellprofiler/jdk/jre/lib/amd64/server JAVA_HOME=/usr/cellprofiler/jdk /usr/cellprofiler/bin/python setup.py install --root=/)
RUN (cd /usr/cellprofiler/src/CellProfiler/cellprofiler/cpmath && PATH=/usr/cellprofiler/bin:$PATH /usr/cellprofiler/bin/python setup.py install --root=/)

RUN git --git-dir=/usr/cellprofiler/src/CellProfiler/.git log -n 1 --format="import datetime; print 'version_string = \"%%s %%s\"' %% (datetime.datetime.utcfromtimestamp(float(%ct)).isoformat('T'), '%h')"| /usr/cellprofiler/bin/python > /usr/cellprofiler/src/CellProfiler/cellprofiler/frozen_version.py
RUN ls -l /usr/cellprofiler/src/CellProfiler/cellprofiler/frozen_version.py
RUN cat /usr/cellprofiler/src/CellProfiler/cellprofiler/frozen_version.py

RUN cp /usr/cellprofiler/src/CellProfiler/usr-bin-cellprofiler /usr/bin/cellprofiler
RUN chmod 755 /usr/bin/cellprofiler


RUN useradd cellprofiler
USER cellprofiler
ENV HOME /home/cellprofiler
WORKDIR /home/cellprofiler

ENTRYPOINT ["/usr/bin/cellprofiler", "-r", "-c"]
CMD ["-h"]
# Use -p filename for execution
