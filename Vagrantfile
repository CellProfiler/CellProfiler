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
      sudo -E apt-get -yq update

      sudo -E apt-get -yq                                       \
        --force-yes                                             \
        --no-install-recommends                                 \
        --no-install-suggests                                   \
          install                                               \
            build-essential                                     \
            curl                                                \
            git                                                 \
            libblas-dev                                         \
            libbz2-dev                                          \
            libfreetype6-dev                                    \
            libhdf5-dev                                         \
            liblapacke-dev                                      \
            libncurses5-dev                                     \
            libpng-dev                                          \
            libreadline-dev                                     \
            libsqlite3-dev                                      \
            libssl-dev                                          \
            libxml2-dev                                         \
            libxslt1-dev                                        \
            llvm                                                \
            openjdk-7-jdk                                       \
            pkg-config                                          \
            wget                                                \
            zlib1g-dev

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

      curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
    SHELL
  end
end
