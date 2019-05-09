FROM nvidia/cuda:10.1-runtime-centos7

LABEL name="RIFT (CentOs)" \
      maintainer="James Alexander Clark <james.clark@ligo.org>" \
      date="20190415" \
      support="nvidia/cuda image"

## RHEL/CentOS 7 64-Bit ##
RUN curl -O http://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm && \
      rpm -ivh epel-release-latest-7.noarch.rpm

#
# Yum-installable dependencies
#
RUN yum update -y && \
      yum install -y vim \
      git \
      python-devel \
      gsl-devel \
      gcc-c++ \
      make \
      cuda-libraries-dev-10-1 \
      cuda-cublas-dev-10-1 \
      cuda-runtime-10-1 \
      cuda-nvcc-10-1 

# RIFT LSCSoft and python dependencies
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py
RUN pip --no-cache-dir install --ignore-installed -U setuptools \
           pip \
           cupy  \
           h5py \
           numba \
           vegas \
           corner \
           scikit-learn \
           healpy \
           matplotlib \
           lalsuite \
           gwdatafind \
           ligo-segments \
           python-ligo-lw 
#           ligo.skymap
RUN CFLAGS='-std=c99' pip --no-cache-dir install -U gwsurrogate 

# RIFT
# Modules and scripts run directly from repository
ENV INSTALL_DIR /opt/lscsoft/rift
ENV ILE_DIR ${INSTALL_DIR}/MonteCarloMarginalizeCode/Code
ENV PATH ${PATH}:${ILE_DIR}
ENV PYTHONPATH ${PYTHONPATH}:${ILE_DIR}
ENV GW_SURROGATE gwsurrogate
RUN git clone https://git.ligo.org/richard-oshaughnessy/research-projects-RIT.git ${INSTALL_DIR} \
      && cd ${INSTALL_DIR} \
      && git checkout temp-RIT-Tides-port_master-GPUIntegration 

# Setup directories for singulary bindings
RUN mkdir -p /cvmfs /hdfs /hadoop /etc/condor
