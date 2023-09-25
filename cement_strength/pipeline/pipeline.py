import os,sys
from cement_strength.exception import CementstrengthException
from cement_strength.logger import logging
from cement_strength.component.data_ingestion import DataIngestion
from cement_strength.config.configuration import Configuration
from cement_strength.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,ModelTrainerArtifact,ModelEvulationArtifact
from cement_strength.component.data_validation import DataValidation
from cement_strength.component.data_transform import DataTransformation
from cement_strength.component.model_trainer import ModelTrainer
from cement_strength.component.model_evalution import Modelevalution

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
        
    def start_data_validation(self,data_ingestion_artifact : DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise CementstrengthException(e,sys) from e
    
    def start_data_transformation(self,data_ingestion_artifact:DataIngestionArtifact,
                                  data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(data_transform_config=self.config.get_data_transformation_config(),
                                                     data_ingestion_artifact=data_ingestion_artifact,
                                                     data_validation_artifact=data_validation_artifact)
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        
    def start_model_trainer(self,data_transform_artifact:DataTransformationArtifact
                             )->ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(model_trainer_config=self.config.get_model_trainer_config(),
                                         data_transform_artifact=data_transform_artifact)
            return model_trainer.intiate_model_trainer()
        except Exception as e:
            raise CementstrengthException(e,sys) from e
    
    def start_model_evalution(self,data_transform_artifact:DataTransformationArtifact,
                              data_validation_artifact:DataValidationArtifact,
                              model_trainer_artifact:ModelTrainerArtifact)->ModelEvulationArtifact:
        try:
            model_evalution = Modelevalution(model_evalution_config=self.config.get_model_evalution_config(),
                                             data_transform_artifact=data_transform_artifact,
                                             data_validation_artifact=data_validation_artifact,
                                             model_trainer_artifact=model_trainer_artifact)
            return model_evalution.initate_model_evulation()
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        
        
    def run_pipeline(self):
        data_igestion_artifact = self.start_data_ingestion()
        data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_igestion_artifact)
        data_transform_artifact = self.start_data_transformation(data_ingestion_artifact=data_igestion_artifact,
                                                             data_validation_artifact=data_validation_artifact)
        model_trainer_artifact = self.start_model_trainer(data_transform_artifact=data_transform_artifact)
        model_evalution_artifact = self.start_model_evalution(data_transform_artifact=data_transform_artifact,
                                                              data_validation_artifact=data_validation_artifact,
                                                              model_trainer_artifact=model_trainer_artifact)