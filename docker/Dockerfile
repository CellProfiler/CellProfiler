FROM ubuntu:18.04

ARG version=4.2.2

RUN apt update
RUN apt -y upgrade
RUN apt install -y make gcc build-essential libgtk-3-dev wget git
RUN apt install -y python3.8-dev python3.8-venv python3-pip openjdk-11-jdk-headless default-libmysqlclient-dev libnotify-dev libsdl2-dev libwebkit2gtk-4.0-dev

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV VIRTUAL_ENV=/opt/venv
RUN python3.8 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install wheel cython numpy

RUN pip install  -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-18.04 wxPython==4.1.0

RUN pip install cellprofiler==$version

CMD ["--run", "--run-headless", "--help"]
