FROM nvidia/cuda:11.2.2-runtime-rockylinux8

LABEL name="RIFT (CentOs) - benchmarking utility" \
  maintainer="James Alexander Clark <james.clark@ligo.org>" \
  date="20220907" \
  support="nvidia/cuda image"
 

## Rocky Linux 8  ##

RUN dnf install --nodocs -y dnf-plugins-core && \
      dnf config-manager --set-enabled powertools && \
      dnf install --nodocs -y epel-release && \
      dnf update -y && \
      dnf install --nodocs -y \
      gcc-c++ \
      git \
      gsl-devel \
      make \
      pandoc \
      python39 \
      python39-devel \
      python39-pip \ 
      python39-tkinter \
      suitesparse \
      suitesparse-devel \
      wget \
      which && \
      dnf clean all && \
      rm -rf /var/cache/dnf

# Pip dependencies
# locking pypandoc so gwsurrogate will build, since it uses ancient version: see https://github.com/man-group/pytest-plugins/issues/87
RUN python3.9 -m pip --no-cache-dir install -U pip==23.0.1 setuptools  && \
      python3.9 -m pip --no-cache-dir install pygsl_lite==0.1.2 && \ 
      python3.9 -m pip --no-cache-dir install -U \
      cupy-cuda11x==11.2.0 \
      vegas \
      healpy \
      cython \
      pypandoc==1.7.5 \
      pyseobnr

RUN  CFLAGS='-std=c99' python3.9 -m pip --no-cache-dir install -U gwsurrogate 
RUN python3.9 -c "import gwsurrogate; gwsurrogate.catalog.pull('NRHybSur3dq8')"
RUN python3.9 -c "import gwsurrogate; gwsurrogate.catalog.pull('NRSur7dq4')"
RUN python3.9 -m pip install -U NRSur7dq2 

# install current master, to test current build
# https://stackoverflow.com/questions/20101834/pip-install-from-git-repo-branch
RUN python3.9 -m pip install -U git+https://git.ligo.org/rapidpe-rift/rift.git@master

# Pin lalsuite to latest SCCB release, currently 7.15
#RUN python3.9 -m pip install lalsuite==7.15
# Development release: use latest pypi
RUN python3.9 -m pip install lalsuite

# Needs numpy to finish first:
ENV GW_SURROGATE=/usr/local/lib64/python3.9/site-packages/gwsurrogate
ENV CUPY_CACHE_IN_MEMORY=1

# Directories we may want to bind
#RUN mkdir -p /ceph /cvmfs /hdfs /hadoop /etc/condor /test
                                                
RUN ln -s /usr/bin/python3 /usr/bin/python

# Environment setup
#COPY entrypoint/bashrc /root/.bashrc
#COPY entrypoint/docker-entrypoint.sh /docker-entrypoint.sh
#ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/bin/bash"]
