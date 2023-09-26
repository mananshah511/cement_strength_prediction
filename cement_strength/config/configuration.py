from cement_strength.exception import CementstrengthException
from cement_strength.logger import logging
from cement_strength.constant import *
import os,sys
from cement_strength.util.util import read_yaml
from cement_strength.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvulationConfig,ModelPusherConfig


class Configuration:

    def __init__(self,
                 config_file_path:str =  CONFIG_FILE_PATH,
                 current_time_stamp:str = CURRENT_TIME_STAMP) -> None:
        try:
            self.config_info = read_yaml(file_path=config_file_path)
            self.time_stamp = current_time_stamp
            self.training_pipeline_config = self.get_training_pipeline_config()
        except Exception as e:
            raise CementstrengthException(e,sys) from e

    def get_data_ingestion_config(self)->DataIngestionConfig:
        try:
            logging.info("get data ingestion config function started")
            artifact_dir = self.training_pipeline_config.artifact_dir

            data_ingestion_config = self.config_info[DATA_INGESTION_CONFIG_KEY]

            data_ingestion_artifact_dir = os.path.join(artifact_dir, DATA_INGESTION_DIR , self.time_stamp)

            data_download_url = data_ingestion_config[DATA_INGESTION_DATASET_DOWNLOAD_URL_KEY]

            raw_data_dir = os.path.join(data_ingestion_artifact_dir, data_ingestion_config[DATA_INGESTION_RAW_DATA_DIR_KEY])

            zip_data_dir = os.path.join(data_ingestion_artifact_dir, data_ingestion_config[DATA_INGESTION_ZIP_DATA_DIR_KEY])

            ingested_data_dir = os.path.join(data_ingestion_artifact_dir, data_ingestion_config[DATA_INGESTION_INGESTED_DATA_DIR_KEY])

            ingested_train_data_dir = os.path.join(ingested_data_dir,data_ingestion_config[DATA_INGESTION_TRAIN_DATA_DIR_KEY])

            ingested_test_data_dir = os.path.join(ingested_data_dir,data_ingestion_config[DATA_INGESTION_TEST_DATA_DIR_KEY])

            data_ingestion_config = DataIngestionConfig(dataset_download_url=data_download_url,
                                                        raw_data_dir=raw_data_dir,
                                                        zip_data_dir=zip_data_dir,
                                                        ingested_train_data_dir=ingested_train_data_dir,
                                                        ingested_test_data_dir=ingested_test_data_dir)
            logging.info(f"data ingestion config : {data_ingestion_config}")
            return data_ingestion_config
        except Exception as e:
            raise CementstrengthException(e,sys) from e

    def get_data_validation_config(self)->DataValidationConfig:
        try:
            logging.info("get data validation function started")
            artifact_dir = self.training_pipeline_config.artifact_dir

            data_validation_config = self.config_info[DATA_VALIDATION_CONFIG_KEY]

            data_validation_artifact_dir = os.path.join(artifact_dir,DATA_VALIDATION_DIR,self.time_stamp)

            schema_file_dir = os.path.join(ROOT_DIR,data_validation_config[DATA_VALIDATION_SCHEMA_DIR_KEY],data_validation_config[DATA_VALIDATION_SCHEMA_FILE_KEY])

            report_file_page_dir = os.path.join(data_validation_artifact_dir)

            report_name = data_validation_config[DATA_VALIDATION_REPORT_PAGE_FILE_KEY]

            data_validation_config = DataValidationConfig(schema_file_dir=schema_file_dir,report_page_file_dir=report_file_page_dir
                                                          ,report_name=report_name)
            logging.info(f"data validation config : {data_validation_config}")

            return data_validation_config

        except Exception as e:
            raise CementstrengthException(e,sys) from e
        
    def get_data_transformation_config(self)->DataTransformationConfig:
        try:
            logging.info("get data transform config function started")
            artifact_dir = self.training_pipeline_config.artifact_dir

            data_transform_config = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]

            data_transform_artifact_dir = os.path.join(artifact_dir,DATA_TRANSFORMATION_DIR,self.time_stamp)

            transform_train_dir = os.path.join(data_transform_artifact_dir,data_transform_config[DATA_TRANSFORMATION_TRAIN_DIR_KEY])
            transform_test_dir = os.path.join(data_transform_artifact_dir,data_transform_config[DATA_TRANSFORMATION_TEST_DIR_KEY])

            transform_graph_dir = os.path.join(data_transform_artifact_dir,data_transform_config[DATA_TRANSFORMATION_GRAPH_DIR_KEY])

            transform_preprocessed_object_dir = os.path.join(data_transform_artifact_dir,data_transform_config[DATA_TRANSFORMATION_PREPROCESSSED_OBJECT_DIR_KEY],
                                                                                                               data_transform_config[DATA_TRANSFORMATION_PREPROCESSED_OBJECT_FILE_NAME_KEY])
            
            cluster_model_dir = os.path.join(data_transform_artifact_dir,data_transform_config[DATA_TRANSFORMATION_CLUSTER_MODEL_DIR_KEY],
                                                                                               data_transform_config[DATA_TRANSFORMATION_CLUSTER_MODEL_FILE_NAME_KEY])
            

            data_transform_config = DataTransformationConfig(transform_train_dir=transform_train_dir,
                                                             transform_test_dir=transform_test_dir,
                                                             graph_save_dir=transform_graph_dir,
                                                             preprocessed_file_path=transform_preprocessed_object_dir,
                                                             cluser_model_file_path=cluster_model_dir)
            logging.info(f"data tranform config: {data_transform_config}")

            return data_transform_config
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        
    def get_model_trainer_config(self)->ModelTrainerConfig:
        try:
            logging.info(f"get model trainer config function started")
            artifact_dir = self.training_pipeline_config.artifact_dir

            model_trainer_config = self.config_info[MODEL_TRAINER_CONFIG_KEY]

            model_trainer_config_dir = os.path.join(artifact_dir,MODEL_TRAINER_DIR,self.time_stamp)

            base_accuracy = model_trainer_config[MODEL_TRAINER_BASE_ACCURACY_KEY]

            config_file_path = os.path.join(ROOT_DIR,model_trainer_config[MODEL_TRAINER_CONFIG_DIR_KEY],
                                            model_trainer_config[MODEL_TRAINER_CONFIG_FILE_NAME_KEY])
            
            model_file_path = os.path.join(model_trainer_config_dir,model_trainer_config[MODEL_TRAINER_MODEL_FILE_NAME_KEY])


            model_trainer_config = ModelTrainerConfig(model_config_file_path=config_file_path,
                                                      trained_model_file_path=model_file_path,
                                                      base_accuracy=base_accuracy)
            
            logging.info(f"model trainer config : {model_trainer_config}")
            
            return model_trainer_config
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        
    def get_model_evalution_config(self)->ModelEvulationConfig:
        try:
            logging.info("get model evalution config function started")
            artifact_dir = self.training_pipeline_config.artifact_dir

            model_evulation_config = self.config_info[MODEL_EVALUTION_CONFIG_KEY]

            model_evulation_file_path = os.path.join(artifact_dir,MODEL_EVALUTION_DIR,
                                                     model_evulation_config[MODEL_EVALUTION_FILE_NAME_KEY])
            model_evulation_config = ModelEvulationConfig(evulation_file_path=model_evulation_file_path,
                                                            time_stamp=self.time_stamp)
                                                            
            logging.info(f"model evulation config : {model_evulation_config}")
            return model_evulation_config
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        
    def get_model_pusher_config(self)->ModelPusherConfig:
        try:
            logging.info(f"get model pusher config function started")
            artifact_dir = self.training_pipeline_config.artifact_dir

            model_pusher_config = self.config_info[MODEL_PUSHER_CONFIG_KEY]

            export_dir_path = os.path.join(artifact_dir,MODEL_PUSHER_DIR,model_pusher_config[MODEL_PUSHER_EXPORT_DIR_KEY]
                                           )
            
            model_pusher_config = ModelPusherConfig(export_dir_path=export_dir_path)

            logging.info(f"model pusher config : {model_pusher_config}")

            return model_pusher_config
        except Exception as e:
            raise CementstrengthException(e,sys) from e

    def get_training_pipeline_config(self)->TrainingPipelineConfig:
        try:
            logging.info("get training pipeline config function started")
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(ROOT_DIR, training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
                                        training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY])
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)
            logging.info(f"training pipeline config : {training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        
