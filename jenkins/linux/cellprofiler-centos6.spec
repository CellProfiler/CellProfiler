% define
pkgname
cellprofiler
% define
pyversion
2.7
% define
tarname
cellprofiler
% define
pref / usr / cellprofiler
% define
python % {pref} / bin / python
% define
githuburl
git + git: // github.com / CellProfiler
% define
javabridge_version
1.0
.9
% define
bioformats_version
1.0
.3
% define
cython_version
0.20
.2

Name: % {pkgname}
Summary:   Cell
image
analysis
software
Version: % {version}
Release: % {release}
Source0: % {tarname}.tar.gz
License:   GPLv2
URL:       http: // www.cellprofiler.org /
Packager:  Vebjorn
Ljosa < ljosa @ broad.mit.edu >
BuildRoot: % {_tmppath} / % {pkgname} - buildroot
Prefix: % {pref}
Requires:  cellprofiler - cython = % {cython_version}
cellprofiler - python
Requires:  cellprofiler - ilastik
cellprofiler - decorator
cellprofiler - h5py
Requires:  cellprofiler - matplotlib
cellprofiler - mysqlpython
cellprofiler - scipy
Requires:  cellprofiler - scikit - learn
cellprofiler - pysqlite
Requires:  cellprofiler - setuptools
cellprofiler - wxpython
cellprofiler - pyzmq
Requires:  cellprofiler - jdk
cellprofiler - pil
xorg - x11 - fonts - Type1
Requires:  liberation - fonts - common
liberation - sans - fonts
Requires:  cellprofiler - javabridge = % {javabridge_version}
Requires:  cellprofiler - bioformats = % {bioformats_version}
Requires:  xorg - x11 - server - Xvfb
cellprofiler - nose
cellprofiler - cellh5
BuildRequires: gcc
gcc - c + + cellprofiler - numpy - devel
cellprofiler - cython
BuildRequires: cellprofiler - python
cellprofiler - ilastik
cellprofiler - decorator
BuildRequires: cellprofiler - h5py
cellprofiler - matplotlib
BuildRequires: cellprofiler - mysqlpython
cellprofiler - scipy
BuildRequires: cellprofiler - pysqlite
cellprofiler - setuptools
BuildRequires: cellprofiler - wxpython
cellprofiler - pyzmq
cellprofiler - jdk
BuildRequires: cellprofiler - javabridge = % {javabridge_version}
BuildRequires: cellprofiler - nose
cellprofiler - bioformats = % {bioformats_version}
BuildRequires: cellprofiler - cellh5

% description
Cell
image
analysis
software

% prep

% setup - q - n
CellProfiler

% build

PATH = % {pref} / bin: % {pref} / jdk / bin:$PATH \
    LD_LIBRARY_PATH = % {pref} / jdk / lib: % {
                                              pref} / jdk / jre / lib / amd64 / server: \
    JAVA_HOME = % {pref} / jdk \
    MAVEN_OPTS = "-Xmx1024m" \
    python
external_dependencies.py

PATH = % {pref} / bin: % {pref} / jdk / bin:$PATH \
    LD_LIBRARY_PATH = % {pref} / jdk / lib: % {
                                              pref} / jdk / jre / lib / amd64 / server: \
    JAVA_HOME = % {pref} / jdk \
    python
CellProfiler.py - -build - and -exit - -do -
not -fetch

patch << EOF
--- CellProfiler.py.orig
2013 - 10 - 16
20:59:07.459360385 - 0400
+++ CellProfiler.py
2013 - 10 - 16
20:16:34.079360393 - 0400


@ @


-20, 6 + 20, 8 @ @
import tempfile
from cStringIO import StringIO

+sys.frozen = True
+
if sys.platform.startswith('win'):
# This recipe is largely from zmq which seems to need this magic
# in order to import in frozen mode - a topic the developers never
EOF

% install

mkdir - p $RPM_BUILD_ROOT % {pref} / src
cp - a. $RPM_BUILD_ROOT % {pref} / src / CellProfiler

PATH = % {pref} / bin: % {pref} / jdk / bin:$PATH \
    LD_LIBRARY_PATH = % {pref} / jdk / lib: % {
                                              pref} / jdk / jre / lib / amd64 / server: \
    JAVA_HOME = % {pref} / jdk \
                  % {python}
cellprofiler / utilities / setup.py
install - -root =$RPM_BUILD_ROOT

PATH = % {pref} / bin: % {pref} / jdk / bin:$PATH \
    LD_LIBRARY_PATH = % {pref} / jdk / lib: % {
                                              pref} / jdk / jre / lib / amd64 / server: \
    JAVA_HOME = % {pref} / jdk \
                  % {python}
cellprofiler / cpmath / setup.py
install - -root =$RPM_BUILD_ROOT

# echo "version_string = '`date +%%Y-%%m-%%dT%%H:%%M:%%S` %{version}'" > $RPM_BUILD_ROOT%{pref}/src/CellProfiler/cellprofiler/frozen_version.py
cd $RPM_BUILD_ROOT % {pref} / src / CellProfiler
PYTHONPATH =../ jenkins / linux / make_frozen_version.py)

mkdir $RPM_BUILD_ROOT / usr / bin
cp
usr - bin - cellprofiler $RPM_BUILD_ROOT / usr / bin / cellprofiler
chmod
755 $RPM_BUILD_ROOT / usr / bin / cellprofiler

% clean

["$RPM_BUILD_ROOT" != "/"] & & rm - rf $RPM_BUILD_ROOT

% files
% defattr(-, root, root)
% {pref} / src / CellProfiler
% {pref} / lib / python % {pyversion} / site - packages / _convex_hull.so
% {pref} / lib / python % {pyversion} / site - packages / _cpmorphology.so
% {pref} / lib / python % {pyversion} / site - packages / _cpmorphology2.so
% {pref} / lib / python % {pyversion} / site - packages / _filter.so
% {pref} / lib / python % {pyversion} / site - packages / _fastemd.so
% {pref} / lib / python % {pyversion} / site - packages / _lapjv.so
% {pref} / lib / python % {pyversion} / site - packages / _propagate.so
% {pref} / lib / python % {pyversion} / site - packages / _watershed.so
% {pref} / lib / python % {pyversion} / site - packages / cpmath - 0.0
.0 - py2
.7.egg - info
% {pref} / lib / python % {pyversion} / site - packages / utilities - 0.0
.0 - py2
.7.egg - info
/ usr / bin / cellprofiler
