from collections import namedtuple

DataIngestionConfig = namedtuple("DataIngestionConfig",
                                 ["dataset_download_url","raw_data_dir","zip_data_dir","ingested_train_data_dir","ingested_test_data_dir"])

DataValidationConfig = namedtuple("DataValidationConfig",
                                ["schema_file_dir","report_page_file_dir","report_name"])

DataTransformationConfig = namedtuple("DataTransformationConfig",
                            ["graph_save_dir","transform_train_dir","transform_test_dir","preprocessed_file_path","cluser_model_file_path"])

ModelTrainerConfig = namedtuple("ModelTrainerConfig",
                                ["trained_model_file_path","base_accuracy","model_config_file_path"])

ModelEvulationConfig = namedtuple("ModelEvulationConfig",
                                  ["evulation_file_path","time_stamp"])


TrainingPipelineConfig = namedtuple("TrainingPipelineConfig",
                                    ["artifact_dir"])
