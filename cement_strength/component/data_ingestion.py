import os,sys
from cement_strength.exception import CementstrengthException
from cement_strength.logger import logging
from cement_strength.entity.artifact_entity import DataIngestionArtifact
from cement_strength.entity.config_entity import DataIngestionConfig
from six.moves import urllib
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class DataIngestion:

    def __init__(self,data_ingestion_config : DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        
    def download_cement_data(self)->str:
        try:
            data_download_url = self.data_ingestion_config.dataset_download_url
            zip_data_dir = self.data_ingestion_config.zip_data_dir

            os.makedirs(zip_data_dir,exist_ok=True)
            file_name = os.path.basename(data_download_url)
            file_name = file_name.replace("+","_")

            zip_file_path = os.path.join(zip_data_dir,file_name)
            logging.info(f"downloading data from url: {data_download_url}")
            logging.info(f"downloading data in {zip_file_path} folder")
            urllib.request.urlretrieve(data_download_url,zip_file_path)
            logging.info("data download completed")
            return zip_file_path
            
        except Exception as e:
            raise CementstrengthException(e,sys) from e

    def exctract_downloaded_data(self,zip_file_path:str):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)

            os.makedirs(raw_data_dir,exist_ok=True)
            logging.info(f"exctrating zip file into : {raw_data_dir}")
            with ZipFile(zip_file_path,'r') as zip:
                zip.extractall(raw_data_dir)
            logging.info("exctraction completed")
        except Exception as e:
            raise CementstrengthException(e,sys) from e

    def split_data_in_train_test(self):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            data_file_name = os.listdir(raw_data_dir)[0]
            logging.info(f"data file name is : {data_file_name}")

            data_file_path = os.path.join(raw_data_dir,data_file_name)

            cement_df = pd.read_excel(data_file_path)

            logging.info("splitting data into train and test files")

            X_train,X_test,y_train,y_test = train_test_split(cement_df.iloc[:,:-1],cement_df.iloc[:,1],test_size=0.2, random_state=42)

            train_df = None
            test_df = None
            

            logging.info(f"combining splitted data to make train and test data")
            train_df = pd.concat([X_train,y_train],axis=1)
            test_df = pd.concat([X_test,y_test],axis=1)

            data_file_name = data_file_name.replace("xls","csv")
            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_data_dir,data_file_name)
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_data_dir,data_file_name)

            logging.info(f"train file path is : {train_file_path}")
            logging.info(f"test file path is : {test_file_path}")

            if train_df is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_data_dir)
                logging.info(f"moving train data as csv")
                train_df.to_csv(train_file_path,index=False)

            if test_df is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_data_dir)
                logging.info(f"moving test data as csv")
                test_df.to_csv(test_file_path,index=False)
            data_ingestion_artifact = DataIngestionArtifact(test_file_path=test_file_path,train_file_path=train_file_path,
                                                        message="Data Ingestion completed successfully",is_ingested=True)
            return data_ingestion_artifact
        except Exception as e:
            raise CementstrengthException(e,sys) from e

    def intiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            zip_file_path = self.download_cement_data()
            raw_file_path = self.exctract_downloaded_data(zip_file_path=zip_file_path)
            return self.split_data_in_train_test()
        except Exception as e:
            raise CementstrengthException(e,sys) from e
    
    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")

