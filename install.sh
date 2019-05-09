#!/bin/bash

# --- Initial Configuration: check me! --- #
INSTALL_DIR=${HOME}/opt/lscsoft/rift
LIGO_USER_NAME="james.clark"
LIGO_ACCOUNTING="ligo.dev.o3.cbc.pe.lalinferencerapid"
GIT_HASH=$(git rev-parse HEAD)

#  Create a virtualenv
virtualenv ${INSTALL_DIR}
pip install -r requirements.txt

# Set up an environment script
ENV_SCRIPT=${INSTALL_DIR}/etc/rift-user-env.sh

mkdir ${INSTALL_DIR}/etc

echo "# ${GIT_HASH}" > ${ENV_SCRIPT}
echo "export LIGO_USER_NAME=${LIGO_USER_NAME}" >> ${ENV_SCRIPT}
echo "export LIGO_ACCOUNTING=${LIGO_ACCOUNTING}" >> ${ENV_SCRIPT}
echo "INSTALL_DIR=${INSTALL_DIR}" >> ${ENV_SCRIPT}
echo "ILE_DIR=\${INSTALL_DIR}/MonteCarloMarginalizeCode/Code" >> ${ENV_SCRIPT}
echo "PATH=${PATH}:${ILE_DIR}" >> ${ENV_SCRIPT}
echo "PYTHONPATH=${PYTHONPATH}:${ILE_DIR}" >> ${ENV_SCRIPT}
echo "GW_SURROGATE=''" >> ${ENV_SCRIPT}

# Source this when virtualenv is activated
echo "source ${ENV_SCRIPT}" >> ${INSTALL_DIR}/bin/activate

# RIFT
git clone https://git.ligo.org/richard-oshaughnessy/research-projects-RIT.git ${INSTALL_DIR}
pushd ${INSTALL_DIR} 
git checkout temp-RIT-Tides-port_master-GPUIntegration
popd

