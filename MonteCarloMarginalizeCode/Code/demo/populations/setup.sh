export OMP_NUM_THREADS=1
export PATH=${PATH}:`pwd`
export LIGO_USER_NAME=muhammad.zeeshan
export LIGO_ACCOUNTING=ligo.dev.o4.cbc.pe.rift
export RIFT_REQUIRE_GPUS='(DeviceName=!="Tesla K10.G1.8GB")&&(DeviceName=!="Tesla K10.G2.8GB")'
export RIFT_AVOID_HOSTS=`cat /home/richard.oshaughnessy/igwn_feedback/rift_avoid_hosts.txt  | tr '\n' , | head -c -1`
export RIFT_GETENV='LD_LIBRARY_PATH,PATH,PYTHONPATH,*RIFT*,LIBRARY_PATH'
export RIFT_GETENV_OSG='*RIFT*,NUMBA_CACHE_DIR'
export NUMBA_CACHE_DIR=/tmp  # needs to be passed to OSG
export SINGULARITY_BASE_EXE_DIR=/usr/local/bin/
export SINGULARITY_RIFT_IMAGE=osdf:///igwn/cit/staging/richard.oshaughnessy/rift_containers/rift_container_ros_seobnr_rift17p3.sif
