import sys
import os

from ChurnPrediction.exception.exception import ChurnPredictionException
from ChurnPrediction.logging.logger import logging
from ChurnPrediction.pipeline.training_pipeline import TrainingPipeline
from ChurnPrediction.entity.config_entity import TrainingPipelineConfig

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd
from fastapi.templating import Jinja2Templates
from ChurnPrediction.utils.main_utils import data_transformations, read_yaml_file

from ChurnPrediction.constant.training_pipeline import TARGET_LABEL,DATA_TRANSFORMATION_COLUMNS_TO_FLOAT64, DATA_TRANSFORMATION_COLUMNS_TO_STR_FOR_BINARY_DECISION
from ChurnPrediction.constant.training_pipeline import DATA_TRANSFORMATION_ROWS_DROPPED_BASEDS_ON_NULL_VALUES_OF_COLUMN,DATA_TRANSFORMATION_COLUMNS_TO_DROP,SCHEMA_FILE_PATH

from ChurnPrediction.utils.main_utils import load_object

templates = Jinja2Templates(directory="./templates")

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        training_pipeline_config = TrainingPipelineConfig()
        train_pipeline = TrainingPipeline(training_pipeline_config=training_pipeline_config)
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise ChurnPredictionException(e,sys) from e
    
@app.post("/predict")
async def predict_route(request: Request,file: UploadFile = File(...)):
    try:
        df=pd.read_csv(file.file)
        #print(df)
        preprocessor=load_object("final_model/preprocessor.pkl")
        final_model=load_object("final_model/LightGBM_final_model/model.pkl")
        df=data_transformations(df,DATA_TRANSFORMATION_COLUMNS_TO_FLOAT64,DATA_TRANSFORMATION_ROWS_DROPPED_BASEDS_ON_NULL_VALUES_OF_COLUMN,DATA_TRANSFORMATION_COLUMNS_TO_STR_FOR_BINARY_DECISION,DATA_TRANSFORMATION_COLUMNS_TO_DROP)
        transformed_data=preprocessor.transform(df)
        feature_names = preprocessor.named_steps['transformer'].get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names)
        transformed_df.columns = transformed_df.columns.str.replace(r"^(num__|cat__)", "", regex=True)
        transformed_df = transformed_df[read_yaml_file(SCHEMA_FILE_PATH,'columns_selected')]
        y_pred = final_model.predict(transformed_df)
        #print(df.iloc[0])
        #y_pred = ChurnPredictionModel.predict(df)
        print(y_pred)
        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        #df['predicted_column'].replace(-1, 0)
        #return df.to_json()
        df.to_csv('prediction_output/output.csv')
        table_html = df.to_html(classes='table table-striped')
        #print(table_html)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
        
    except Exception as e:
            raise ChurnPredictionException(e,sys) from e

    
if __name__=="__main__":
    app_run(app,host="0.0.0.0",port=8000)