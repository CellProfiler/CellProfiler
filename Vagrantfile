# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure(2) do |configuration|
  configuration.vm.define :ubuntu, primary: true do |ubuntu|
    ubuntu.vm.box = "hashicorp/precise64"

    ubuntu.vm.provision "shell", privileged: false, inline: <<-SHELL
      sudo add-apt-repository ppa:chris-lea/zeromq

      sudo -E apt-get -yq update

      sudo -E apt-get -yq                                      \
        --force-yes                                            \
        --no-install-recommends                                \
        --no-install-suggests                                  \
          install                                              \
            build-essential                                    \
            default-jdk                                        \
            openjdk-7-jdk                                      \
            git                                                \
            libxml2-dev                                        \
            libxslt1-dev                                       \
            python-dev                                         \
            python-nose                                        \
            python-pip                                         \
            python-software-properties                         \
            python-virtualenv                                  \
            software-properties-common

      source                                                   \
        $HOME/.bashrc                                       && \
      [ -z "$JAVA_HOME" ]                                   && \
      echo                                                     \
        "export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64" >> $HOME/.bashrc

      source $HOME/.bashrc

      mkdir -p $HOME/virtualenv

      virtualenv                                               \
        --system-site-packages                                 \
          $HOME/virtualenv/python2.7_with_system_site_packages

      cd /vagrant

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
            python-numpy                                       \
            python-pandas                                      \
            python-scipy                                       \
            python-tk                                          \
            python-wxgtk2.8                                    \
            python-zmq

      source $HOME/virtualenv/python2.7_with_system_site_packages/bin/activate

      pip install -U pip wheel
      pip install -r requirements.txt
      pip install -U javabridge
      pip install -e git+https://github.com/CellH5/cellh5.git#egg=cellh5

      python                                                   \
          CellProfiler.py                                      \
        --build-and-exit

      # python                                                   \
      #     cpnose.py                                            \
      #   --noguitests                                           \
      #     cellprofiler/cpmath/tests

      # python                                                   \
      #     cpnose.py                                            \
      #   --noguitests                                           \
      #     cellprofiler/matlab/tests

      # python                                                   \
      #     cpnose.py                                            \
      #   --noguitests                                           \
      #   --with-javabridge                                      \
      #     cellprofiler/modules/tests

      python                                                   \
          cpnose.py                                            \
        --noguitests                                           \
        --with-javabridge                                      \
          cellprofiler/tests

      # python                                                   \
      #     cpnose.py                                            \
      #   --noguitests                                           \
      #   --with-javabridge                                      \
      #     cellprofiler/utilities/tests
    SHELL
  end
end
