


## Virtual environments (for dependencies)
The prerequisites for this code may not be available, can take a while to install, and (if used without care) may impact your other workflows via pip user installls.
To simplify safe sharing, make a virtual environment which holds these dependencies.

```
mkdir virtualenvs
virtualenv virtualenvs/rapidpe_gpu_clean
source virtualenvs/rapidpe_gpu_clean/bin/activate
cd ${ILE_DIR}
pip install matploblib==2.2.4  # fixme, should be in setup
python setup.py install 
pip install cupy   # must run on GPU-enabled machine
```

## Containers (for OSG)
