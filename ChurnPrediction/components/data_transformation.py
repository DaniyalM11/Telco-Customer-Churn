import sys
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ChurnPrediction.constant.training_pipeline import TARGET_LABEL,DATA_TRANSFORMATION_COLUMNS_TO_FLOAT64, DATA_TRANSFORMATION_COLUMNS_TO_STR_FOR_BINARY_DECISION
from ChurnPrediction.constant.training_pipeline import DATA_TRANSFORMATION_ROWS_DROPPED_BASEDS_ON_NULL_VALUES_OF_COLUMN,DATA_TRANSFORMATION_COLUMNS_TO_DROP,SCHEMA_FILE_PATH


from ChurnPrediction.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from ChurnPrediction.entity.config_entity import DataTransformationConfig
from ChurnPrediction.exception.exception import ChurnPredictionException
from ChurnPrediction.logging.logger import logging
from ChurnPrediction.utils.main_utils import save_numpy_array_data, save_object
from ChurnPrediction.utils.main_utils import read_yaml_file,write_yaml_file, read_csv_data


class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact = data_validation_artifact
            self.data_transformation_config:DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise ChurnPredictionException(e,sys) from e     

    @classmethod
    def get_data_transformer_object(cls, num_cols: list, cat_cols: list) -> Pipeline:
        """
        Initializes a transformation pipeline with the following steps:
        1. StandardScaler for scaling numerical columns.
        2. OneHotEncoder for encoding categorical columns.

        Args:
          cls: DataTransformation class
          num_cols: List of numerical feature column names.
          cat_cols: List of categorical feature column names.

        Returns:
          A ColumnTransformer object wrapped inside a Pipeline.
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )
        try:
            # Standard Scaler for numerical columns
            scaler: StandardScaler = StandardScaler()

            # OneHotEncoder for categorical columns
            one_hot_encoder: OneHotEncoder = OneHotEncoder(
                drop="first", sparse_output=False
            )

            # ColumnTransformer to handle different feature types
            column_transformer: ColumnTransformer = ColumnTransformer(
                transformers=[
                    ("num", scaler, num_cols),  # Apply StandardScaler to numerical columns
                    ("cat", one_hot_encoder, cat_cols),  # Apply OneHotEncoder to categorical columns
                ]
            )

            # Create a pipeline with the transformer
            processor: Pipeline = Pipeline([
                ("transformer", column_transformer)  # Apply scaling and encoding
            ])

            logging.info("Initialized transformation pipeline successfully.")
            return processor

        except Exception as e:
            logging.error("Error occurred in get_data_transformer_object method.")
            raise ChurnPredictionException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info(
            "Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
        except Exception as e:
            raise ChurnPredictionException(e, sys) from e    

    @staticmethod
    def checkVIF(X) -> pd.DataFrame:
        vif = pd.DataFrame()
        vif['Features'] = X.columns
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['VIF'] = round(vif['VIF'], 2)
        vif = vif.sort_values(by = "VIF", ascending = False,)
        return(vif)
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info(
            "Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df = read_csv_data(self.data_validation_artifact.valid_train_file_path)
            test_df = read_csv_data(self.data_validation_artifact.valid_test_file_path)

            for df in [train_df, test_df]:
                for item in DATA_TRANSFORMATION_COLUMNS_TO_FLOAT64:
                    df[item] = pd.to_numeric(df[item], errors='coerce').astype('float64')

            for df in [train_df, test_df]:
                for item in DATA_TRANSFORMATION_ROWS_DROPPED_BASEDS_ON_NULL_VALUES_OF_COLUMN:
                    df.dropna(subset=[item], inplace = True)

            for df in [train_df, test_df]:
                for item in DATA_TRANSFORMATION_COLUMNS_TO_STR_FOR_BINARY_DECISION:
                    df[item] = df[item].astype(str)
                    df[item] = df[item].apply(lambda x: 'Yes' if x == '1' else 'No')

            for df in [train_df, test_df]:
                df.drop(columns=DATA_TRANSFORMATION_COLUMNS_TO_DROP, inplace=True)     

            for df in [train_df, test_df]:
                df[TARGET_LABEL] = df[TARGET_LABEL].apply(lambda x: '1' if x == 'Yes' else '0')
                df[TARGET_LABEL] = df[TARGET_LABEL].astype(int)     
            
            # training DataFrame
            X_train = train_df.drop(columns=[TARGET_LABEL],axis=1)
            y_train = train_df[TARGET_LABEL].reset_index(drop=True)
            #print(X_train.columns)
            #print(y_train)
            #print(X_train.head())
            #X_train.to_csv(r"C:\Users\Daniy\Telco Customer Churn Prediction\testing2.csv", index=False,header=True)
            # testing DataFrame
            X_test = test_df.drop(columns=[TARGET_LABEL],axis=1)
            y_test = test_df[TARGET_LABEL].reset_index(drop=True)
            #print(X_test.columns)
            #print(y_test)   
            #print(X_train.head())
            #X_test.to_csv(r"C:\Users\Daniy\Telco Customer Churn Prediction\testing3.csv", index=False,header=True)

            
            cat_cols = [col for col in X_train.columns if (df[col].dtype == 'object')]
            print(cat_cols)
            num_cols = [col for col in X_train.columns if col not in cat_cols]
            print(num_cols)

            #################################################################################################        
            # preprocessor for training data
            preprocessor = self.get_data_transformer_object(
                num_cols=num_cols,cat_cols=cat_cols
                )
            # fit and transform training and test data
            preprocessor_object = preprocessor.fit(X_train)
            transformed_x_train = preprocessor_object.transform(X_train)
            transformed_x_test = preprocessor_object.transform(X_test)
            #print(type(transformed_x_train))
            #print(type(transformed_x_test))
            ##########################################################################################################

            # One Hot Encoding
            encoder = OneHotEncoder(drop='first', sparse_output=False)
            encoder.fit(X_train[cat_cols])

            X_train_cat_ohe = encoder.transform(X_train[cat_cols])
            X_test_cat_ohe = encoder.transform(X_test[cat_cols])
            cat_ohe_cols = encoder.get_feature_names_out(cat_cols)

            # Standard Scaling for Numerical Features
            scaler = StandardScaler()
            scaler.fit(X_train[num_cols])

            X_train_num_scaled = pd.DataFrame(scaler.transform(X_train[num_cols]), columns=num_cols)
            X_test_num_scaled = pd.DataFrame(scaler.transform(X_test[num_cols]), columns=num_cols)

            # Concatenating Scaled Numerical and Encoded Categorical Features
            transformed_x_train_df = pd.concat([
                X_train_num_scaled.reset_index(drop=True),  
                pd.DataFrame(X_train_cat_ohe, columns=cat_ohe_cols)  
            ], axis=1)

            transformed_x_test_df = pd.concat([
                X_test_num_scaled.reset_index(drop=True), 
                pd.DataFrame(X_test_cat_ohe, columns=cat_ohe_cols)  
            ], axis=1)

            #transformed_x_train.to_csv(r"C:\Users\Daniy\Telco Customer Churn Prediction\testing.csv", index=False,header=True)

                        
            #apply VIF
            vif_report_before_df = self.checkVIF(transformed_x_train_df)
            write_yaml_file(self.data_transformation_config.VIF_report_before_dir, vif_report_before_df.to_dict(), replace=True)
            vif_report_after_df = self.checkVIF(pd.DataFrame(transformed_x_train_df)[read_yaml_file(SCHEMA_FILE_PATH,'columns_selected')])
            write_yaml_file(self.data_transformation_config.VIF_report_after_dir, vif_report_after_df.to_dict(), replace=True)
            
            dir_path=os.path.dirname(self.data_transformation_config.transformed_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            transformed_x_train_df = transformed_x_train_df[read_yaml_file(SCHEMA_FILE_PATH,'columns_selected')]
            transformed_x_test_df = transformed_x_test_df[read_yaml_file(SCHEMA_FILE_PATH,'columns_selected')]
            
            concat_train = pd.concat([transformed_x_train_df, y_train],axis=1)
            concat_test = pd.concat([transformed_x_test_df, y_test],axis=1)

            concat_train.to_csv(self.data_transformation_config.transformed_train_file_path, index=False, header=True)
            concat_test.to_csv(self.data_transformation_config.transformed_test_file_path, index=False, header=True)

            #train_arr = np.c_[transformed_x_train, np.array(y_train)]
            #test_arr = np.c_[transformed_x_test, np.array(y_test)]

            #save numpy array data
            #save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr)
            #save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr)
            #save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            save_object( "final_model/preprocessor.pkl", preprocessor_object)

            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            raise ChurnPredictionException(e, sys) from e                    





