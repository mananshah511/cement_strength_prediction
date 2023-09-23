import os,sys
from cement_strength.exception import CementstrengthException
from cement_strength.logger import logging
from cement_strength.entity.config_entity import DataTransformationConfig
from cement_strength.entity.artifact_entity import DataTransformationArtifact,DataIngestionArtifact,DataValidationArtifact
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from cement_strength.util.util import read_yaml
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans
import csv
from pathlib import Path
import dill

class DataTransformation:

    def __init__(self,data_transform_config:DataTransformationConfig,
                 data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_artifact:DataValidationArtifact) -> None:
        try:
            logging.info(f"{'>>'*20}Data Transformation log started.{'<<'*20} ")
            self.data_transform_config = data_transform_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise CementstrengthException(e,sys) from e
    
    @staticmethod
    def log_transformation(X):
        try:
            logging.info(f"log transformation function started")
            return np.log(X+1)
        except Exception as e:
            raise CementstrengthException(e,sys) from e

    def get_pre_processing_object(self)->Pipeline:
        try:
            logging.info(f"get preprocessing object function started")

            logging.info(f"making transformer object for the log transform")
            transformer = FunctionTransformer(DataTransformation.log_transformation)

            logging.info(f"pre processing pipeline esamble started")
            preprocessingobj = Pipeline(steps=[('log_transformer',transformer),
                                       ('scaler',StandardScaler())])
            logging.info(f"pre processing pipeline esamble completed : {preprocessingobj}")
            return preprocessingobj
            
            
        except Exception as e:
            raise CementstrengthException(e,sys) from e   
         
    def perform_pre_processing(self, pre_processing_object:Pipeline, is_test_data:bool = False)->pd.DataFrame:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path
            schema_file = read_yaml(file_path=schema_file_path)
            target_column_name = schema_file['target_column']
            if is_test_data == False:
                
                train_file_path = self.data_ingestion_artifact.train_file_path
                train_df = pd.read_csv(train_file_path)
            
                target_df = train_df.iloc[:,-1]
                
                train_df.drop(target_column_name,axis=1,inplace=True)
                
                columns = list(train_df.columns)
                train_df = pre_processing_object.fit_transform(train_df)
               
                train_df = pd.DataFrame(train_df,columns=columns)
                
                train_df = pd.concat([train_df , target_df],axis=1)
               
                return train_df
            else:
              
                test_file_path = self.data_ingestion_artifact.test_file_path
                test_df = pd.read_csv(test_file_path)
                target_df = test_df.iloc[:,-1]
                test_df.drop(target_column_name,axis=1,inplace=True)
                columns = list(test_df.columns)
                test_df = pre_processing_object.transform(test_df)
                test_df = pd.DataFrame(test_df,columns=columns)
                test_df = pd.concat([test_df , target_df],axis=1)
                return test_df
        except Exception as e:
            raise CementstrengthException(e,sys) from e 
        
    def get_and_save_cluster_graph(self, train_df):
        try:
            logging.info(f"finding scheme path for target columns name")
            schema_file_path = self.data_validation_artifact.schema_file_path
            logging.info(f"scheme file path is : {schema_file_path}")
            schema_file = read_yaml(file_path=schema_file_path)
            target_column_name = schema_file['target_column']
            logging.info(f"target columns name is : {target_column_name}")

            kmeans = KMeans(init='k-means++',random_state=42)
            visualizer = KElbowVisualizer(kmeans, k=(1,11))
            visualizer.fit((train_df.drop(target_column_name,axis=1))) 
            os.makedirs(self.data_transform_config.graph_save_dir,exist_ok=True)
            visualizer.show(self.data_transform_config.graph_save_dir+"\graph_cluser.png")
           
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        
    def get_and_save_silhouette_score_graph(self,train_df):
        try:
            
            for no_of_culsters in [2,3,4,5]:

                kmeans = KMeans(n_clusters=no_of_culsters, init='k-means++',random_state=42)
                visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
                visualizer.fit(train_df)
                os.makedirs(self.data_transform_config.graph_save_dir,exist_ok=True) 
                visualizer.show(self.data_transform_config.graph_save_dir+"\cluster_"+str(no_of_culsters)+"_silihoetter_score.png")
                #1 bug active graphs are overlapping
                

        except Exception as e:
            raise CementstrengthException(e,sys) from e
    def save_data_based_on_cluster(self,n_clusters,train_df,test_df):
        logging.info(f"save data based on cluster function started")

        logging.info(f"finding scheme path for target columns name")
        schema_file_path = self.data_validation_artifact.schema_file_path
        logging.info(f"scheme file path is : {schema_file_path}")
        schema_file = read_yaml(file_path=schema_file_path)
        target_column_name = schema_file['target_column']
        logging.info(f"target columns name is : {target_column_name}")


        logging.info(f"cluster object intilized and data fit is completed")
        kmeans = KMeans(n_clusters=3, init='k-means++',random_state=42)
        kmeans.fit(train_df.drop(target_column_name,axis=1))

        logging.info(f"predicting clusters mapping to data based on k-mean clusters for train data")
        train_predict = kmeans.predict((train_df.drop(target_column_name,axis=1)))
        logging.info(f"getting transform train data path and making dir")
        transform_train_path = self.data_transform_config.transform_train_dir
        os.makedirs(transform_train_path,exist_ok=True)

        
        columns_name = list(train_df.columns)
        logging.info(f"columns names are : {columns_name}")
        cluster_numbers = list(np.unique(np.array(train_predict)))
        logging.info(f"cluster numbers are : {cluster_numbers}")

        logging.info(f"making csv files cluster wise for training data")
        for cluster_number in cluster_numbers:
            file_path = os.path.join(transform_train_path,'train_cluster'+str(cluster_number)+'.csv')
            with Path(file_path).open('w',newline='') as csvfiles:
                csvwriter = csv.writer(csvfiles)

                csvwriter.writerow(columns_name)
                for index in range(len(train_predict)):
                    if train_predict[index] == cluster_numbers[cluster_number]:
                        csvwriter.writerow(train_df.iloc[index])
        logging.info(f"csv file writing for training data is successfull")

        logging.info(f"predicting clusters mapping to data based on k-mean clusters for test data")
        test_predict = kmeans.predict((test_df.drop(target_column_name,axis=1)))
        logging.info(f"getting transform test data path and making dir")
        transform_test_path = self.data_transform_config.transform_test_dir
        os.makedirs(transform_test_path,exist_ok=True)

        
        columns_name = list(test_df.columns)
        logging.info(f"columns names are : {columns_name}")
        cluster_numbers = list(np.unique(np.array(test_predict)))
        logging.info(f"cluster numbers are : {cluster_numbers}")

        logging.info(f"making csv files cluster wise for testing data")
        for cluster_number in cluster_numbers:
            file_path = os.path.join(transform_test_path,'test_cluster'+str(cluster_number)+'.csv')
            with Path(file_path).open('w',newline='') as csvfiles:
                csvwriter = csv.writer(csvfiles)

                csvwriter.writerow(columns_name)
                for index in range(len(test_predict)):
                    if test_predict[index] == cluster_numbers[cluster_number]:
                        csvwriter.writerow(train_df.iloc[index])
        logging.info(f"csv file writing for testing data is successfull")


    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            preprocessed_obj = self.get_pre_processing_object()
            pre_dir_name = os.path.dirname(self.data_transform_config.preprocessed_file_path)
            os.makedirs(pre_dir_name,exist_ok=True)
            with open(self.data_transform_config.preprocessed_file_path,'wb') as pre_file:
                dill.dump(preprocessed_obj,pre_file)
            logging.info(f"pre-processing object saved")
            train_df = self.perform_pre_processing(pre_processing_object=preprocessed_obj)
      
            test_df = self.perform_pre_processing(pre_processing_object=preprocessed_obj,is_test_data=True)
         
            self.get_and_save_cluster_graph(train_df=train_df)
            self.get_and_save_silhouette_score_graph(train_df=train_df)
            self.save_data_based_on_cluster(n_clusters=3,train_df=train_df,test_df=test_df)
            data_transform_artifact = DataTransformationArtifact(transform_test_dir=self.data_transform_config.transform_test_dir,
                                                                 transform_train_dir=self.data_transform_config.transform_train_dir,
                                                                 preprocessed_object_dir=self.data_transform_config.preprocessed_file_path,
                                                                 message="Data transformation completed successfully",
                                                                 is_transfromed=True)
            return data_transform_artifact

        except Exception as e:
            raise CementstrengthException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*20}Data Transformation log completed.{'<<'*20} \n\n")
