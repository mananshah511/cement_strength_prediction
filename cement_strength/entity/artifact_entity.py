from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact",["train_file_path","test_file_path",
                                                            "message","is_ingested"])

DataValidationArtifact = namedtuple("DataValidationArtifact",["schema_file_path","report_file_path",
                                                              "message","is_validated"])


