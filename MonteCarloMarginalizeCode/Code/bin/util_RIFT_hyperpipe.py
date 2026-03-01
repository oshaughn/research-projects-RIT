#! /usr/bin/env python

# ARGUMENT SPECIFICATION
#   arch
#     method:  <string>
#     n-iterations
#     n-samples-per-job
#     explode marg jobs: 
#   post:
#     coord-module: <string>
#     coords-fit:
#     coords-sample:
#     exe:
#     settings:
#        n-max
#        n-step
#        sampler-method
#        sigma-cut
#   marg-list:
#      - name:
#        exe:
#        args:
#      - name:
#        ....
#    puff:
#       exe:
#       settings:
#          puff-factor: 
#    test:
#    init:
#       file:
#       generation:
#            placement method: <string> 
#            params_and_ranges:
#            npts:
#            external_code:
#            external arguments:
#    general:
#      retries
#      request-disk
#      request-memory
#      use-osg
#    
#

import numpy as np
import os

# install reuires hydra-core
from omegaconf import DictConfig, OmegaConf
import hydra



@hydra.main(version_base=None, config_path=".",config_name='hyperpipe_config')
def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
