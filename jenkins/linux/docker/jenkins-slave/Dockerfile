# BUILD-USING: docker build -t jenkins-slave /root/CellProfiler/jenkins/linux/docker/jenkins-slave
# RUN-USING: docker run -d -p 2222:22 -v /root/CellProfiler:/var/jenkins/workspace/CellProfiler_Linux:rw --name=jenkins-slave jenkins-slave
FROM ubuntu
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 36A1D7869245C8950F966E92D8576A8BA88D21E9
RUN echo deb http://get.docker.io/ubuntu docker main> /etc/apt/sources.list.d/docker.list
RUN apt-get update
RUN apt-get install -y openssh-server openjdk-6-jdk lxc-docker
RUN mkdir /var/run/sshd
RUN mkdir /root/.ssh
RUN chmod 700 /root/.ssh
ADD id_rsa.pub /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys
CMD ["/usr/sbin/sshd", "-D"]
