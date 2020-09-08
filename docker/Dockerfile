FROM ubuntu:16.04

ARG version=4.0.1

# Install CellProfiler dependencies
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt -y update
RUN apt -y upgrade
RUN apt install -y python3.8-dev python3.8-distutils python3-pip openjdk-8-jdk-headless libmysqlclient-dev libnotify-dev libsdl2-dev wget

# Set up environment
RUN export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
RUN export PATH=$PATH:/home/ubuntu/.local/bin

# This is needed for a Python 3.8 quirk
RUN apt remove -y python3-pip
RUN python3.8 -m easy_install pip

#Install wx from a wheel
RUN wget https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04/wxPython-4.1.0-cp38-cp38-linux_x86_64.whl
RUN python3.8 -m pip install wxPython-4.1.0-cp38-cp38-linux_x86_64.whl
RUN rm wxPython-4.1.0-cp38-cp38-linux_x86_64.whl

# Install CellProfiler
RUN python3.8 -m pip install cellprofiler==$version

WORKDIR /usr/local/src
ENTRYPOINT ["cellprofiler"]

CMD ["--run", "--run-headless", "--help"]
