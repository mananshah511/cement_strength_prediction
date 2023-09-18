import os,sys
from cement_strength.exception import CementstrengthException
from cement_strength.logger import logging
from cement_strength.entity.artifact_entity import DataIngestionArtifact
from cement_strength.entity.config_entity import DataIngestionConfig


class DataIngestion:

    def __init__(self,data_ingestion_config : DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        
    def download_data(self):
        pass

