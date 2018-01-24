sudo apt-get -y update

sudo apt-get -y upgrade

sudo apt-get -y install    \
    build-essential     \
    cython              \
    git                 \
    libmysqlclient-dev  \
    libhdf5-dev         \
    libxml2-dev         \
    libxslt1-dev        \
    openjdk-8-jdk       \
    python-dev          \
    python-pip          \
    python-h5py         \
    python-matplotlib   \
    python-mysqldb      \
    python-scipy        \
    python-numpy        \
    python-wxgtk3.0     \
    python-zmq

if ! cd $HOME/CellProfiler; then
    git clone https://github.com/CellProfiler/CellProfiler.git $HOME/CellProfiler
fi

sudo -H pip install --upgrade

sudo -H pip install --editable $HOME/CellProfiler
