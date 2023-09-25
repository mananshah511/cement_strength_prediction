import os
from datetime import datetime

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

ROOT_DIR = os.getcwd()
CONFIG_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR ,CONFIG_DIR ,CONFIG_FILE_NAME)

CURRENT_TIME_STAMP = get_current_time_stamp()

NO_CLUSTER = 4

#training pipeline related variables

TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"

#data ingestion related variables

DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_DATASET_DOWNLOAD_URL_KEY = "dataset_download_url"
DATA_INGESTION_DIR = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_ZIP_DATA_DIR_KEY = "zip_data_dir"
DATA_INGESTION_INGESTED_DATA_DIR_KEY = "ingested_dir"
DATA_INGESTION_TRAIN_DATA_DIR_KEY = "ingested_train_dir"
DATA_INGESTION_TEST_DATA_DIR_KEY = "ingested_test_dir"

#data validation related varibales

DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_DIR = "data_validation"
DATA_VALIDATION_SCHEMA_DIR_KEY = "scheme_dir"
DATA_VALIDATION_SCHEMA_FILE_KEY = "schema_file"
DATA_VALIDATION_REPORT_PAGE_FILE_KEY = "report_page_file_name"

#data transformation related variables

DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_DIR = "data_transformation"
DATA_TRANSFORMATION_GRAPH_DIR_KEY = "graph_save_dir"
DATA_TRANSFORMATION_TRAIN_DIR_KEY = "train_dir"
DATA_TRANSFORMATION_TEST_DIR_KEY = "test_dir"
DATA_TRANSFORMATION_PREPROCESSSED_OBJECT_DIR_KEY = "preprocessed_object_dir"
DATA_TRANSFORMATION_PREPROCESSED_OBJECT_FILE_NAME_KEY = "preprocessed_object_file_name"
DATA_TRANSFORMATION_CLUSTER_MODEL_DIR_KEY = "cluster_model_dir"
DATA_TRANSFORMATION_CLUSTER_MODEL_FILE_NAME_KEY = "cluster_model_file_name"

#model trainer related variables

MODEL_TRAINER_CONFIG_KEY = "model_trainer_config"
MODEL_TRAINER_DIR = "model_trainer"
MODEL_TRAINER_BASE_ACCURACY_KEY = "base_accuracy"
MODEL_TRAINER_MODEL_FILE_NAME_KEY = "model_file_name"
MODEL_TRAINER_CONFIG_DIR_KEY = "model_config_dir"
MODEL_TRAINER_CONFIG_FILE_NAME_KEY = "model_config_file_name"


#model evalution related variable

MODEL_EVALUTION_CONFIG_KEY = "model_evalution_config"
MODEL_EVALUTION_DIR = "model_evalution"
MODEL_EVALUTION_FILE_NAME_KEY = "model_evulation_file_name"


BEST_MODEL_KEY = "best_model"
HISTORY_KEY = "history"
MODEL_PATH_KEY = "model_path"








