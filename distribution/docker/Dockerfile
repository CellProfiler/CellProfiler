# must be amd64, wxPython wheel for arm64 is not available
FROM amd64/ubuntu:20.04

# bypass apt installation frontend interaction
ARG DEBIAN_FRONTEND=noninteractive

ARG cp_version=4.2.3

RUN apt update
RUN apt -y upgrade
RUN apt install -y make gcc build-essential libgtk-3-dev wget git
RUN apt install -y python3.9-dev python3.9-venv python3-pip openjdk-11-jdk-headless default-libmysqlclient-dev libnotify-dev libsdl2-dev libwebkit2gtk-4.0-dev

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV VIRTUAL_ENV=/opt/venv
RUN python3.9 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install wheel cython numpy

RUN pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-20.04/wxPython-4.2.1-cp39-cp39-linux_x86_64.whl

RUN pip install cellprofiler==$cp_version

WORKDIR /usr/local/src
ENTRYPOINT ["cellprofiler"]

CMD ["--run", "--run-headless", "--help"]
