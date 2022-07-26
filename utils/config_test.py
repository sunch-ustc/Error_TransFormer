import os
import yaml 
from utils.utils import mkdir_if_missing
# import sys
# sys.path.append('../') 
def create_config(  config_file_exp):
    # Config for environment path 
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream) 
    cfg = {}  
    # Copy
    for k, v in config.items():
        cfg[k] = v 
  
    return cfg 
