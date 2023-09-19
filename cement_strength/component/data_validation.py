import os,sys
from cement_strength.exception import CementstrengthException
from cement_strength.logger import logging
from cement_strength.entity.config_entity import DataValidationConfig
from cement_strength.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
import pandas as pd
import numpy as np
from cement_strength.util.util import read_yaml
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

class DataValidation:

    def __init__(self, data_validation_config : DataValidationConfig,
                 data_ingestion_artifact : DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20}Data Validation log started.{'<<'*20} ")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        
    def get_train_test_dataframe(self):
        try:
            logging.info(f"get train test dataframe function started")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"reading succesfull")
            return train_df,test_df
        
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        
    def check_train_test_dir_exist(self)->bool:
        try:
            logging.info(f"check train and test dir function started")

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            flag_train = False
            flag_test = False

            if os.path.exists(train_file_path):
                flag_train = True
                logging.info(f"train dir is available")

            if os.path.exists(test_file_path):
                flag_test = True
                logging.info(f"test dir is available")

            if flag_train == False:
                logging.info(f"train dir is not available")
                return False
            if flag_test == False:
                logging.info(f"test dir is not available")
                return False
            
            return flag_test and flag_train

        except Exception as e:
            raise CementstrengthException(e,sys) from e
    def check_column_count_validation(self)->bool:
        try:
            logging.info(f"check columns count validation function started")
            scheme_file_path = self.data_validation_config.schema_file_dir
            scheme_file = read_yaml(file_path=scheme_file_path)

            train_df,test_df = self.get_train_test_dataframe()

            train_column_count = len(train_df.columns)
            test_column_count = len(test_df.columns)

            schema_column_count = len(scheme_file['columns'])

            logging.info(f"number of columns in train file is : {train_column_count}")
            logging.info(f"number of columns in test file is : {test_column_count}")

            logging.info(f"number of columns as per schema file is : {schema_column_count}")

            train_flag = False
            test_flag = False

            if schema_column_count == train_column_count:
                train_flag = True
                logging.info(f"number of columns in train file is correct")

            if schema_column_count == test_column_count:
                test_flag = True
                logging.info(f"number of columns in test file is correct")

            if train_flag == False:
                logging.info(f"number of columns in train file is not correct")
                return False
            
            if test_flag == False:
                logging.info(f"number of columns in test file is not correct")
                return False
            
            return train_flag and test_flag
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        

    def check_column_name_validation(self)->bool:
        try:
            logging.info(f"check columns name validation function started")
            scheme_file_path = self.data_validation_config.schema_file_dir
            scheme_file = read_yaml(file_path=scheme_file_path)

            train_df,test_df = self.get_train_test_dataframe()

            schema_column_name_list = scheme_file['numerical_columns']
            logging.info(f"schema columns list is : {schema_column_name_list}")

            train_column_name_list = list(train_df.columns)
            test_column_name_list = list(test_df.columns)

            logging.info(f"train column list is : {train_column_name_list}")
            logging.info(f"test column list is : {test_column_name_list}")

            train_flag = False
            test_flag = False

            if schema_column_name_list == train_column_name_list:
                train_flag = True
                logging.info(f"train columns are correct")

            if schema_column_name_list == test_column_name_list:
                test_flag = True
                logging.info(f"test columns are correct")

            if train_flag == False:
                logging.info(f"columns in train file is not correct")
                return False
            
            if test_flag == False:
                logging.info(f"columns in test file is not correct")
                return False
            
            return train_flag and test_flag

        except Exception as e:
            raise CementstrengthException(e,sys) from e

    def check_column_data_type_validation(self)->bool:
        try:
            logging.info(f"check columns data type validation function started")
            scheme_file_path = self.data_validation_config.schema_file_dir
            scheme_file = read_yaml(file_path=scheme_file_path)

            train_df,test_df = self.get_train_test_dataframe()

            train_columns_data = dict(train_df.dtypes)
            test_columns_data = dict(test_df.dtypes)

            schema_columns_data = scheme_file['columns']

            logging.info(f"training file data type info : {train_columns_data}")
            logging.info(f"test file data type info : {test_columns_data}")

            logging.info(f"schema file data type info : {schema_columns_data}")

            train_flag = False
            test_flag = False

            for column_name in schema_columns_data.keys():
                if schema_columns_data[column_name] != train_columns_data[column_name]:
                    logging.info(f"data type for {column_name} in train file is not correct")
                    return train_flag
                if schema_columns_data[column_name] != test_columns_data[column_name]:
                    logging.info(f"data type for {column_name} in test file is not correct")
                    return test_flag
            logging.info("data type for train file is correct")
            logging.info("data type for test file is correct")
            train_flag=True
            test_flag=True

            return train_flag and test_flag
        except Exception as e:
            raise CementstrengthException(e,sys) from e
                    

    def check_null_in_columns(self)->bool:
        try:
            logging.info(f"check null value in columns validation started")
            train_df,test_df = self.get_train_test_dataframe()

            train_null_value_count = dict(train_df.isnull().sum())
            test_null_value_count = dict(test_df.isnull().sum())

            logging.info(f"train file null value count : {train_null_value_count}")
            logging.info(f"test file null value count : {test_null_value_count}")

            for column_name,null_count in train_null_value_count.items():
                if null_count>0:
                    logging.info(f"null count in train file's {column_name} is {null_count}")
                    return False
            
            for column_name,null_count in test_null_value_count.items():
                if null_count>0:
                    logging.info(f"null count in test file's {column_name} is {null_count}")
                    return False
            logging.info(f"no null values found in train file")
            logging.info(f"no null values found in test file")
            return True
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        

    def get_and_save_data_drift_report(self):
        try:
            logging.info(f"get and save data drift report function started")
            report_file_dir = self.data_validation_config.report_page_file_dir
            os.makedirs(report_file_dir, exist_ok=True)
            report_file_path = os.path.join(report_file_dir,self.data_validation_config.report_name)
            dashboard = Dashboard(tabs=[DataDriftTab()])
            train_df,test_df = self.get_train_test_dataframe()
            dashboard.calculate(train_df,test_df)
            dashboard.save(report_file_path)
            logging.info(f"report saved sucessfully")

        except Exception as e:
            raise CementstrengthException(e,sys) from e

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            validation5 = False
            validation1 = self.check_train_test_dir_exist()
            if validation1:
                validation2 = self.check_column_count_validation()
            if validation2:
                validation3 = self.check_column_name_validation()
            if validation3:
                validation4 = self.check_column_data_type_validation()
            if validation4:
                validation5 = self.check_null_in_columns()
            self.get_and_save_data_drift_report()
            

            data_validation_artifact = DataValidationArtifact(schema_file_path=self.data_validation_config.schema_file_dir,
                                                              report_file_path=self.data_validation_config.report_page_file_dir,
                                                              message="Data Validation completed",
                                                              is_validated=validation5)
            return data_validation_artifact
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Data Validation log completed.{'<<'*20} \n\n")