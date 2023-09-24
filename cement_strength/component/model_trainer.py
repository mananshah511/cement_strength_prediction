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
            self.model_trainer_config = model_trainer_config
            self.data_transform_artifact = data_transform_artifact
        except Exception as e:
            raise CementstrengthException(e,sys) from e
        
    def intiate_model_trainer(self)->ModelTrainerArtifact:

        logging.info(f"intiate model trainer function started")

        logging.info("loading train data")
        transform_train_file = self.data_transform_artifact.transform_train_dir
        train_df = pd.read_csv(transform_train_file+"\\train_cluster2.csv")

        logging.info("loading test data")
        transform_test_file = self.data_transform_artifact.transform_test_dir
        test_df = pd.read_csv(transform_test_file+"\\test_cluster2.csv")

        logging.info("splitting data")
        X_train,y_train,X_test,y_test = train_df.iloc[:,:-1],train_df.iloc[:,-1],test_df.iloc[:,:-1],test_df.iloc[:,-1]

        logging.info("exctracting model cofig file path")
        model_config_file_path = self.model_trainer_config.model_config_file_path

        logging.info("intilization of model factory class")
        model_factory = ModelFactory(model_config_path=model_config_file_path)

        base_accuracy = self.model_trainer_config.base_accuracy

        best_model = model_factory.get_best_model(X=np.array(X_train),y=np.array(y_train),base_accuracy=base_accuracy)

        logging.info(f"best model : {best_model}")

        grid_searched_best_model_list:list[GridSearchedBestModel] = model_factory.grid_searchd_best_model_list

        model_list = [model.best_model for model in grid_searched_best_model_list]
        
        metric_info:MetricInfoArtifact=get_evulate_regression_model(X_train=np.array(X_train),y_train=np.array(y_train),
                                                                    X_test=np.array(X_test),
                                                                        y_test=np.array(y_test),base_accuracy=base_accuracy,model_list=model_list)
        
        model_object = metric_info.model_object

        save_model_path = self.model_trainer_config.trained_model_file_path

        dirname = os.path.dirname(save_model_path)

        os.makedirs(dirname)

        with open(save_model_path,'wb') as obj_file:
            dill.dump(model_object,obj_file)

        model_trainer_artifact = ModelTrainerArtifact(is_trained=True,
                                                          message="Model trained successfully",
                                                          trained_model_path=self.model_trainer_config.trained_model_file_path,
                                                          train_rmse=metric_info.train_rmse,
                                                          test_rmse=metric_info.test_rmse,
                                                          train_accuracy=metric_info.train_accuracy,
                                                          test_accuracy=metric_info.test_accuracy,
                                                          model_accuracy=metric_info.model_accuracy)
        
        return model_trainer_artifact
        
