import yaml
from cement_strength.exception import CementstrengthException
import sys
from cement_strength.logger import logging

def read_yaml(file_path:str):
    try:
        with open(file_path,"rb") as yaml_file:
            logging.info(f"Loading yaml file from : {file_path}")
            return yaml.safe_load(yaml_file)
            
    except Exception as e:
        raise CementstrengthException(e,sys) from e
