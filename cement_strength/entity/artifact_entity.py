from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact",["train_file_path","test_file_path",
                                                            "message","is_ingested"])

DataValidationArtifact = namedtuple("DataValidationArtifact",["schema_file_path","report_file_path",
                                                              "message","is_validated"])

DataTransformationArtifact = namedtuple("DataTransformationArtifact",["transform_train_dir",
                                                                      "transform_test_dir","preprocessed_object_dir",
                                                                      "cluster_model_dir",
                                                                      "message","is_transfromed"])
ModelTrainerArtifact = namedtuple("ModelTrainerArtifact",["is_trained","message","train_rmse","test_rmse",
                                                          "train_accuracy","test_accuracy","model_accuracy"])




