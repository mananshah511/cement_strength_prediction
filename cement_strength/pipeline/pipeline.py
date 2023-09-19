import os,sys
from cement_strength.exception import CementstrengthException
from cement_strength.logger import logging
from cement_strength.component.data_ingestion import DataIngestion
from cement_strength.config.configuration import Configuration
from cement_strength.entity.artifact_entity import DataIngestionArtifact

class Pipeline:

    def __init__(self,config:Configuration=Configuration()) -> None:
        try:
            self.config = config
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        
    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.intiate_data_ingestion()
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        
    def run_pipeline(self):
        data_igestion_artifact = self.start_data_ingestion()