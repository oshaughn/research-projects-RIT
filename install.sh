#!/bin/bash

# --- Initial Configuration: check me! --- #
VIRTUALENV_DIR=${HOME}/opt/lscsoft/rift
INSTALL_DIR=${VIRTUALENV_DIR}/research-projects-RIT
LIGO_USER_NAME="james.clark"
LIGO_ACCOUNTING="ligo.dev.o3.cbc.pe.lalinferencerapid"
GIT_HASH=$(git rev-parse HEAD)
# ---------------------------------------- #

#   #  Create a virtualenv
#   virtualenv ${VIRTUALENV_DIR}
#   source ${VIRTUALENV_DIR}/bin/activate && pip install -U --no-cache -r requirements.txt

# Set up an environment script
ENV_SCRIPT=${VIRTUALENV_DIR}/etc/rift-user-env.sh

mkdir -p ${VIRTUALENV_DIR}/etc

echo "# ${GIT_HASH}" > ${ENV_SCRIPT}
echo "export LIGO_USER_NAME=${LIGO_USER_NAME}" >> ${ENV_SCRIPT}
echo "export LIGO_ACCOUNTING=${LIGO_ACCOUNTING}" >> ${ENV_SCRIPT}
echo "export INSTALL_DIR=${INSTALL_DIR}" >> ${ENV_SCRIPT}
echo "export ILE_DIR=${INSTALL_DIR}/MonteCarloMarginalizeCode/Code" >> ${ENV_SCRIPT}
echo "export PATH=${PATH}:${ILE_DIR}" >> ${ENV_SCRIPT}
echo "export PYTHONPATH=${PYTHONPATH}:${ILE_DIR}" >> ${ENV_SCRIPT}
echo "export GW_SURROGATE=''" >> ${ENV_SCRIPT}

# Source this when virtualenv is activated
echo "source ${ENV_SCRIPT}" >> ${VIRTUALENV_DIR}/bin/activate

#   # RIFT
#   git clone https://git.ligo.org/richard-oshaughnessy/research-projects-RIT.git ${INSTALL_DIR}
#   pushd ${INSTALL_DIR} 
#   git checkout temp-RIT-Tides-port_master-GPUIntegration
#   popd
#
#   # Source the environment manually just this once
#   source ${ENV_SCRIPT}
