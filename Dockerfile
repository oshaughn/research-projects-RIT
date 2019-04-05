FROM ligo/builder:el7
#FROM ligo/base:el7

LABEL name="RIFT EL7" \
      maintainer="James Alexander CLark <james.clark@ligo.org>" \
      date="20190405" \
      support="Experimental Platform"


# Setup directories for binding
RUN mkdir -p /cvmfs /hdfs /hadoop /etc/condor /build

RUN yum upgrade -y && \
    yum clean all && \
    rm -rf /var/cache/yum

RUN yum update && yum install -y \
      python-pip git wget \
      && yum clean all \
      && rm -rf /var/cache/yum


RUN wget https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-10.1.105-1.x86_64.rpm \
      && rpm -i cuda-repo-rhel7-10.1.105-1.x86_64.rpm 
      
RUN yum install -y cuda  python-devel \
      && rm -rf /var/cache/yum

RUN pip install -U setuptools pip
RUN pip install cupy lalsuite

ENV export INSTALL_DIR=/opt/lscsoft/rift
ENV ILE_DIR=${INSTALL_DIR}/MonteCarloMarginalizeCode/Code
ENV PATH=${PATH}:${ILE_DIR}
ENV PYTHONPATH=${PYTHONPATH}:${ILE_DIR}

#RUN git clone https://github.com/oshaughn/research-projects-RIT.git
RUN git clone https://git.ligo.org/richard-oshaughnessy/research-projects-RIT.git \
      && cd research-projects-RIT \
      && git checkout temp-RIT-Tides-port_master-GPUIntegration \
      && python setup.py install


