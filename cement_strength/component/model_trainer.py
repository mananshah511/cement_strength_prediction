import os,sys
from cement_strength.exception import CementstrengthException
from cement_strength.logger import logging
from cement_strength.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
from cement_strength.entity.config_entity import ModelTrainerConfig
from cement_strength.entity.model_trainer import ModelFactory,get_evulate_regression_model,GridSearchedBestModel,MetricInfoArtifact
import pandas as pd
import numpy as np
import dill


class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transform_artifact:DataTransformationArtifact) -> None:
        try:
            logging.info(f"{'>>'*20}Model Trainer log started.{'<<'*20} ")
            self.model_trainer_config = model_trainer_config
            self.data_transform_artifact = data_transform_artifact
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        
    def intiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info(f"intiate model trainer function started")

            transform_train_files = self.data_transform_artifact.transform_train_dir
            logging.info(f"transform training files are available at : {transform_train_files}")
            transform_test_files = self.data_transform_artifact.transform_test_dir
            logging.info(f"transform testing files are available at : {transform_test_files}")
            
            train_files = list(os.listdir(transform_train_files))
            test_files = list(os.listdir(transform_test_files))
            logging.info(f"training files are : {train_files}")
            logging.info(f"testing files are : {test_files}")

            model_trainer_artifact = None
            train_rmse = []
            test_rmse = []
            train_accuracy = []
            test_accuracy = []
            model_accuracy = []

            for cluster_number in range(len(train_files)):
                logging.info(f"{'>>'*20}cluster : {cluster_number}{'<<'*20}")

                train_file_name = train_files[cluster_number]
                test_file_name = test_files[cluster_number]

                train_file_path = os.path.join(transform_train_files,train_file_name)
                test_file_path = os.path.join(transform_test_files,test_file_name)

                logging.info(f"reading train data from the file : {train_file_path}")    
                train_df = pd.read_csv(train_file_path)
                logging.info(f"train data reading successfull")
                logging.info(f"reading test data from the file : {test_file_path}")    
                test_df = pd.read_csv(test_file_path)
                logging.info(f"test data reading successfull")

                logging.info("splitting data into input and output feature")
                X_train,y_train,X_test,y_test = train_df.iloc[:,:-1],train_df.iloc[:,-1],test_df.iloc[:,:-1],test_df.iloc[:,-1]

                logging.info("exctracting model cofig file path")
                model_config_file_path = self.model_trainer_config.model_config_file_path
                logging.info("intilization of model factory class")
                model_factory = ModelFactory(model_config_path=model_config_file_path)

                base_accuracy = self.model_trainer_config.base_accuracy
                logging.info(f"base accuracy is : {base_accuracy}")
            
                logging.info(f"finding best model for the cluster : {cluster_number}")
                best_model = model_factory.get_best_model(X=np.array(X_train),y=np.array(y_train),base_accuracy=base_accuracy)
                logging.info(f"best model on trained data is : {best_model}")

                grid_searched_best_model_list:list[GridSearchedBestModel] = model_factory.grid_searchd_best_model_list
                model_list = [model.best_model for model in grid_searched_best_model_list]
                logging.info(f"individual best model list : {model_list}")

                logging.info(f"finding best model after evulation on train and test data")
                metric_info:MetricInfoArtifact=get_evulate_regression_model(X_train=np.array(X_train),y_train=np.array(y_train),
                                                                        X_test=np.array(X_test),
                                                                            y_test=np.array(y_test),base_accuracy=base_accuracy,model_list=model_list)
                

                
                
                
                model_object = metric_info.model_object
                logging.info(f"----------best model after train and test evulation : {model_object} accuracy : {metric_info.model_accuracy}-----------")
                model_path = self.model_trainer_config.trained_model_file_path
                model_base_name = os.path.basename(model_path)
                logging.info(f"base model name is : {model_base_name}")
                dir_name = os.path.dirname(model_path)
                cluster_dir_name = os.path.join(dir_name,'cluster'+str(cluster_number))
                logging.info(f"model will be saved in {cluster_dir_name}")
                os.makedirs(cluster_dir_name,exist_ok=True)
                model_cluster_path = os.path.join(cluster_dir_name,model_base_name)

                with open(model_cluster_path,'wb') as obj_file:
                    dill.dump(model_object,obj_file)
                logging.info(f"model saved successfully")
                #with open('model_info.txt','a') as txtfile:
                #    txtfile.write(f"----------best model after train and test evulation : {model_object} accuracy : {metric_info.model_accuracy}-----------\n")

                train_rmse.append(metric_info.train_rmse)
                test_rmse.append(metric_info.test_rmse)
                train_accuracy.append(metric_info.train_accuracy)
                test_accuracy.append(metric_info.test_accuracy)
                model_accuracy.append(metric_info.model_accuracy)


            model_trainer_artifact = ModelTrainerArtifact(is_trained=True,
                                                        message="Model trained successfully",
                                                        trained_model_path=self.model_trainer_config.trained_model_file_path,
                                                        train_rmse=train_rmse,
                                                        test_rmse=test_rmse,
                                                        train_accuracy=train_accuracy,
                                                        test_accuracy=test_accuracy,
                                                        model_accuracy=model_accuracy)
            return model_trainer_artifact
        except Exception as e:
            raise CementstrengthException(e,sys) from e
    def __del__(self):
        logging.info(f"{'>>'*20}Model trainer log completed.{'<<'*20} \n\n")
        


            

            

            

            

            

            
            
            
            
            
            
            

            

            

            

            
            
            
        
    
    
