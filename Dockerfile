FROM nvidia/cuda:11.7.0-runtime-rockylinux8

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
RUN python3.9 -m pip --no-cache-dir install -U pip setuptools  && \
      CFLAGS='-std=c99' python3.9 -m pip --no-cache-dir install -U gwsurrogate && \
      python3.9 -m pip --no-cache-dir install -U \
      cupy-cuda117 \
      vegas \
      healpy \
      cython \
      pypandoc \
      NRSur7dq2 \
      RIFT==0.0.15.7rc1 && \
      python3.9 -c "import gwsurrogate; gwsurrogate.catalog.pull('NRHybSur3dq8'); gwsurrogate.catalog.pull('NRSur7dq4')"

# Needs numpy to finish first:
ENV GW_SURROGATE=/usr/local/lib64/python3.6/site-packages/gwsurrogate
ENV CUPY_CACHE_IN_MEMORY=1

# Directories we may want to bind
RUN mkdir -p /ceph /cvmfs /hdfs /hadoop /etc/condor /test
                                                
# Assume python3 is used, don't force specific version
#RUN ln -sf /usr/bin/python3.6 /usr/bin/python

# Environment setup
#COPY entrypoint/bashrc /root/.bashrc
#COPY entrypoint/docker-entrypoint.sh /docker-entrypoint.sh
#ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/bin/bash"]
