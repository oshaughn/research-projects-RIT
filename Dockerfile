FROM nvidia/cuda:10.0-runtime-centos7

LABEL name="RIFT (CentOs) - benchmarking utility" \
  maintainer="James Alexander Clark <james.clark@ligo.org>" \
  date="20190926" \
  support="nvidia/cuda image"
 
## RHEL/CentOS 7 64-Bit ##
RUN curl -O http://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm && \
      rpm -ivh epel-release-latest-7.noarch.rpm && \
      rm epel-release-latest-7.noarch.rpm && \
      yum upgrade -y

RUN yum install -y \
      gcc-c++ \
      git \
      gsl-devel \
      make \
      pandoc \
      python3-devel \
      python3-pip \
      tkinter \
      suitesparse \
      suitesparse-devel \
      wget \
      which

RUN yum clean all && \
      rm -rf /var/cache/yum

# Pip dependencies
RUN pip3 --no-cache-dir install -U pip setuptools

# Special dependencies
RUN pip3 --no-cache-dir install \
      cupy-cuda100 \
      vegas \
      healpy \
      cython \
      numpy \
      pypandoc

# Needs numpy to finish first:
RUN CFLAGS='-std=c99' pip3 --no-cache-dir install -U gwsurrogate
ENV GW_SURROGATE=/usr/local/lib64/python3.6/site-packages/gwsurrogate
ENV CUPY_CACHE_IN_MEMORY=1

# RIFT
#RUN pip3 --no-cache-dir install git+https://github.com/oshaughn/research-projects-RIT.git@d14110cdb41fef1adb461932600c3c11a82e4db6
RUN pip3 --no-cache-dir install RIFT==0.0.15.4rc9

# Directories we may want to bind
RUN mkdir -p /ceph /cvmfs /hdfs /hadoop /etc/condor /test
                                                
# Download surrogate data
RUN pip3 --no-cache-dir install NRSur7dq2
RUN python3 -c "import gwsurrogate; gwsurrogate.catalog.pull('NRHybSur3dq8')"
RUN python3 -c "import gwsurrogate; gwsurrogate.catalog.pull('NRSur7dq4')"

# Assume python3 is used, don't force specific version
#RUN ln -sf /usr/bin/python3.6 /usr/bin/python
