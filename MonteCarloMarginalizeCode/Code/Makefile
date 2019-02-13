
build_gpu:
# Default configuration
#   - cuda10
#   - tesla P40/P100  [head nodes] or K10   https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list
#      -arch=sm_60  # ldas-pcdev13 architecture  (GTX 10 series also pascal-powered, should work for all)
# Check configuraito with
#   nvidia-smi   # prints available machines
	nvcc -arch=sm_70 -cubin -o cuda_Q_inner_product.cubin cuda_Q_inner_product.cu 


# AVAILABLE GPU MACHINES (2019-01 edition)
# CIT:
# 560 nodes with single GTX 1050 Ti
# 32 nodes with dual K10 PCI cards (shows up as 4 GPU devices) and dual GTX 750 Ti
# 16 nodes with dual GTX 750 Ti (and dual Xeon Phi co-processor)

# LHO and LLO each:
# 40 nodes with GTX 1050 Ti
# single DGX-1 system with 8 tightly coupled V100

# And all 3 sites advertise the login/head node GPU devices in their MOTD.


# In general, you can use condor_status to query all the machine in a local Condor pool to see what resources are being advertised as available to run jobs. This includes information about GPUs when available. For example, the CIT cluster is currently advertising 872 Condor slots with CUDA 9.2,

# ldas-grid:~> condor_status -af CUDADeviceName -constraint "DynamicSlot =!= True" | sort | uniq -c
#     549 GeForce GTX 1050 Ti
#      16 GeForce GTX 750 Ti
#    1911 undefined

# ldas-grid:~> condor_status -af CUDA0DeviceName -constraint "DynamicSlot =!= True" | sort | uniq -c
#      30 GeForce GTX 750 Ti
#       2 Tesla K10.G2.8GB
#       1 Tesla K80
#    2442 undefined

test:
# test hlm generation, overlaps
#	python test_hlm.py   # shows that hlm's generate via our wrapper.  SHOULD BE TESTED with other approximants.
	python test_rhotimeseries.py  # inner product with data
# test likelihood evaluation
	python test_precompute_singleifo.py
	python test_precompute.py
	python test_precompute_noisydata.py
# test sampler 
	test_mcsamp.py
# test marginalization (single template).  User may need to interact 
	test_like_and_samp_singleifo.py
	test_like_and_samp.py
# test Fisher matrix estimator
	test_effectiveFisher.py



