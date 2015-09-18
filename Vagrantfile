# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure(2) do |configuration|
  configuration.vm.define :ubuntu, primary: true do |ubuntu|
    ubuntu.vm.box = "ubuntu/trusty64"

    ubuntu.vm.provider :virtualbox do |virtualbox|
      virtualbox.cpus = 2

      virtualbox.memory = 8192
    end

    ubuntu.vm.provision "shell", privileged: false, inline: <<-SHELL
      sudo -E apt-get -yq                                       \
        --force-yes                                             \
        --no-install-recommends                                 \
        --no-install-suggests                                   \
          install                                               \
            build-essential                                     \
            cython                                              \
            git                                                 \
            openjdk-7-jdk                                       \
            python-dev                                          \
            python-h5py                                         \
            python-imaging                                      \
            python-lxml                                         \
            python-matplotlib                                   \
            python-mysqldb                                      \
            python-pandas                                       \
            python-pip                                          \
            python-tk                                           \
            python-scipy                                        \
            python-vigra                                        \
            python-wxgtk2.8                                     \
            python-zmq

      source                                                    \
        $HOME/.bashrc                                        && \
      [ -z "$JAVA_HOME" ]                                    && \
      echo                                                      \
        "export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64" >> $HOME/.bashrc

      source                                                    \
        $HOME/.bashrc                                        && \
      [ -z "$LD_LIBRARY_PATH" ]                              && \
      echo                                                      \
        "export LD_LIBRARY_PATH=/usr/lib/jvm/java-7-openjdk-amd64/jre/lib/amd64/server:/usr/lib/jvm/java-7-openjdk-amd64:/usr/lib/jvm/java-7-openjdk-amd64/include" >> $HOME/.bashrc

      sudo pip install --upgrade pip wheel

      sudo pip install --requirement /vagrant/requirements.txt

      sudo pip install --editable git+https://github.com/CellH5/cellh5.git#egg=cellh5

      # python /vagrant/CellProfiler.py --build-and-exit

      # python /vagrant/cpnose.py --noguitests --with-javabridge /vagrant/cellprofiler/cpmath/tests
      # python /vagrant/cpnose.py --noguitests --with-javabridge /vagrant/cellprofiler/matlab/tests
      # python /vagrant/cpnose.py --noguitests --with-javabridge /vagrant/cellprofiler/modules/tests
      # python /vagrant/cpnose.py --noguitests --with-javabridge /vagrant/cellprofiler/tests
      # python /vagrant/cpnose.py --noguitests --with-javabridge /vagrant/cellprofiler/utilities/tests
    SHELL
  end
end
