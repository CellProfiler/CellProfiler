FROM centos:centos6

# See cellprofiler.org/linux.shtml

COPY cellprofiler.repo /etc/yum.repos.d/
RUN yum install -y cellprofiler-2.1.1

RUN useradd cellprofiler
USER cellprofiler
ENV HOME /home/cellprofiler
WORKDIR /home/cellprofiler

ENTRYPOINT ["cellprofiler", "-r", "-c"]
CMD ["-h"]
# Use -p filename for execution
