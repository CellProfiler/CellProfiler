# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure(2) do |configuration|
  configuration.vm.box = "hashicorp/precise64"

  configuration.vm.provision "shell", inline: <<-SHELL
    sudo add-apt-repository ppa:chris-lea/zeromq

    sudo -E apt-get -yq                                      \
      --force-yes                                            \
      --no-install-recommends                                \
      --no-install-suggests                                  \
        install                                              \
          build-essential                                    \
          default-jdk                                        \
          git                                                \
          libxml2-dev                                        \
          libxslt1-dev                                       \
          python-dev                                         \
          python-nose                                        \
          python-pip                                         \
          python-software-properties                         \
          python-virtualenv                                  \
          software-properties-common

    mkdir $HOME/virtualenv

    virtualenv                                               \
      --system-site-packages                                 \
        $HOME/virtualenv/python2.7_with_system_site_packages

    cd /vagrant

    find -E . -regex '.*\.(pyc|so)' -delete

    sudo -E apt-get -yq update

    sudo -E apt-get -yq                                      \
      --force-yes                                            \
      --no-install-recommends                                \
      --no-install-suggests                                  \
        install                                              \
          libhdf5-serial-dev                                 \
          libzmq3-dev                                        \
          python-h5py                                        \
          python-imaging                                     \
          python-lxml                                        \
          python-matplotlib                                  \
          python-mysqldb                                     \
          python-pandas                                      \
          python-scipy                                       \
          python-tk                                          \
          python-wxgtk2.8

    source $HOME/virtualenv/python2.7_with_system_site_packages/bin/activate

    pip install -U pip wheel
    pip install -r requirements.txt
    pip install -e git+https://github.com/CellH5/cellh5.git#egg=cellh5

    python external_dependencies.py -o

    python CellProfiler.py --build-and-exit

    python                                                   \
        cpnose.py                                            \
      --noguitests                                           \
        cellprofiler/cpmath/tests

    python                                                   \
        cpnose.py                                            \
      --noguitests                                           \
        cellprofiler/matlab/tests

    python                                                   \
        cpnose.py                                            \
      --noguitests                                           \
      --with-javabridge                                      \
        cellprofiler/modules/tests

    python                                                   \
        cpnose.py                                            \
      --noguitests                                           \
      --with-javabridge                                      \
        cellprofiler/tests

    python                                                   \
        cpnose.py                                            \
      --noguitests                                           \
      --with-javabridge                                      \
        cellprofiler/utilities/tests
  SHELL
end
