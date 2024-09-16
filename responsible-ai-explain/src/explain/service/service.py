'''
Copyright 2024 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies 
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE 
AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from random import randint
import zipfile
from explain.service.responsible_ai_explain import ResponsibleAIExplain
from explain.config.config import read_config_yaml
from explain.config.logger import CustomLogger, request_id_var
from explain.dao.workbench.FileStoreDb import fileStoreDb
from explain.dao.workbench.Model import Model, ModelAttributes, ModelAttributeValues
from explain.dao.workbench.Dataset import Dataset, DatasetAttributes, DatasetAttributeValues
from explain.dao.workbench.Tenet import Tenet
from explain.dao.workbench.Html import Html
from explain.dao.workbench.Batch import Batch
from explain.dao.workbench.Preprocessor import Preprocessor
from explain.dao.explainability.TblException import Tbl_Exception
from explain.dao.explainability.TblExplanationMethods import Tbl_Explanation_Methods
from explain.exception.exception import MissingPDFException
from explain.utils.report import Report
from explain.utils.create_csv import CreateCSV
from explain.utils.pdf_img_pdf import pdf_to_img, images_to_pdf
from explain.mappers.mappers import ExplainabilityLocalDemoRequest, ExplainabilityGlobalRequest, ExplainabilityGlobalResponse, \
    ExplainabilityLocalTabularRequest, ExplainabilityLocalTabularResponse, ExplainabilityLocalTabular, \
    ExplainOutput, ExplainFileOutput, ExplainabilityLocalResponseDemo, \
    explainabilityglobal, importantFeatures, GetExplanationMethodsResponse, \
    ExplainabilityTabular, GetExplanationResponse, GetReportResponse, ExplainabilityLocalTextRequest, \
    ExplainabilityLocalTextResponse, ExplainabilityLocalText, ExplainabilityLocalTextResponse, ExplainabilityLocalText, ExplainabilityLocalTextResponse, \
    ExplainabilityLocalImageRequest, ExplainabilityLocalImageResponse, \
    ExplainabilityTabular_New, ExplainabilityCoTResponse
from nutanix_object_storage import NutanixObjectStorage
from alibi.utils import visualization
from matplotlib.colors import LinearSegmentedColormap
from io import StringIO, BytesIO
from fastapi.responses import StreamingResponse
from datetime import datetime
from matplotlib import pyplot as plt
from uuid import uuid4
import pandas as pd
import numpy as np 
import shutil
import base64
import joblib
import keras
import time
import json
import os
import cv2
import requests

log = CustomLogger()

class Payload:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class ExplainService:

    def save_as_json_file(fileName:str,content):
        with open(fileName, "w") as outfile:
            json.dump(content,outfile,indent=2)

    def save_as_file(filename:str, content):
        with open(filename,"wb") as outfile:
            outfile.write(content)
    
    def save_html_to_file(html_string, filename):
        with open(filename, 'w') as f:
            f.write(html_string)

    def localText_explain (payload: ExplainabilityLocalTextRequest) -> ExplainabilityLocalTextResponse:
        log.debug(f"payload: {payload}")
        
        try:
            bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(payload.modelPredictionEndpoint)
            model_file_content = NutanixObjectStorage.get_file_content(bucket_dict["bucket_name"],
                                                                       bucket_dict["object_key"])
            
            if bucket_dict["object_key"].split(".")[-1] == "pkl":
                ExplainService.save_as_file(filename="../models/local_explain_text_temp_model.pkl",
                                            content=model_file_content)
                local_explain_model = joblib.load("../models/local_explain_text_temp_model.pkl")
            else:
                with open("../models/local_explain_text_temp_model.h5", "wb") as f:
                    f.write(model_file_content)
                local_explain_model = keras.models.load_model("../models/local_explain_text_temp_model.h5")

            if payload.vectorizerPredictionEndpoint is not None:                
                bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(payload.vectorizerPredictionEndpoint)
                model_file_content = NutanixObjectStorage.get_file_content(bucket_dict["bucket_name"],
                                                                           bucket_dict["object_key"])

                ExplainService.save_as_file(filename="../models/local_explain_text_temp_vectorizer.pkl",
                                            content=model_file_content)

        except Exception as e:
            log.error(e, exc_info=True)
            raise Exception
        # local_explain_model = joblib.load("../models/local_explain_text_temp_model.pkl")

        if payload.vectorizerPredictionEndpoint is not None:
            local_explain_vectorizer = joblib.load("../models/local_explain_text_temp_vectorizer.pkl")
        else:
            local_explain_vectorizer = None

        obj_explain = ResponsibleAIExplain.local_explain_text(text=payload.inputText,
                                                              model=local_explain_model,
                                                              vectorizer=local_explain_vectorizer,
                                                              method=payload.methods,
                                                              segmentationType=payload.segmentationType,
                                                              class_names=payload.targetClassNames
                                                            )
        
        log.debug(f"obj_explain: {obj_explain}")
        
        List_explain_text = []
        for item in obj_explain:
            log.debug(f"item: {item}")
            predictedTarget = item['predictedTarget']
            if item.get('anchor'):
                objexplainabilitylocalTextTemp = ExplainabilityLocalText(predictedTarget=predictedTarget,
                                                                             anchor=item['anchor'],
                                                                             attributions=None)
            elif item.get('attributions'):
                objexplainabilitylocalTextTemp = ExplainabilityLocalText(predictedTarget=predictedTarget,
                                                                             anchor=None,
                                                                             attributions=item['attributions'])
            else:
                objexplainabilitylocalTextTemp = ExplainabilityLocalText(predictedTarget=predictedTarget)
            
            List_explain_text.append(objexplainabilitylocalTextTemp)

        objExplainabilityLocalResponse = ExplainabilityLocalTextResponse(explainerID=payload.explainerID, 
                                                                         explanation=List_explain_text
                                                                         )

        local_file_path = "../output/localText_explain.json"
        ExplainService.save_as_json_file(local_file_path, objExplainabilityLocalResponse.dict())
        bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(payload.outputPath)
        NutanixObjectStorage.upload_with_high_threshold(
            bucket_name=bucket_dict['bucket_name'],
            object_key=bucket_dict['object_key'],
            file_size_mb=10,
            local_file_path=local_file_path
            )
        
        if payload.methods == 'INTEGRATED-GRADIENTS':
            ExplainService.save_as_json_file("../output/localText_explain_IG.html", obj_explain[0]['attributions'])
            
            object_key = "/".join(bucket_dict['object_key'].split("/")[:-1])
            log.debug(f"object_key: {object_key}")
            NutanixObjectStorage.upload_with_high_threshold(
                bucket_name=bucket_dict['bucket_name'],
                object_key=f"{object_key}/localText_explain_IG.html",
                file_size_mb=10,
                local_file_path="../output/localText_explain_IG.html"
            )
        return objExplainabilityLocalResponse
    
    def localImage_explain (payload: ExplainabilityLocalImageRequest) -> ExplainabilityLocalImageResponse:
        log.debug(f"payload: {payload}")

        try:
            bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(payload.modelPredictionEndpoint)
            model_file_content = NutanixObjectStorage.get_file_content(bucket_dict["bucket_name"], 
                                                                       bucket_dict["object_key"]
                                                                       )
            if bucket_dict["object_key"].split(".")[-1] == "pkl":
                ExplainService.save_as_file(filename="../models/local_explain_image_model.pkl",
                                            content=model_file_content)
                local_explain_image_model = joblib.load("../models/local_explain_image_model.pkl")
            else:
                with open("../models/local_explain_image_model.h5", "wb") as f:
                    f.write(model_file_content)
                local_explain_image_model = keras.models.load_model("../models/local_explain_image_model.h5")                                                    

            # ExplainService.save_as_file(filename="../models/local_explain_image_model.pkl",content=model_file_content)

            bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(payload.inputImagePath)
            input_image_content = NutanixObjectStorage.get_file_content(bucket_dict["bucket_name"], 
                                                                        bucket_dict["object_key"]
                                                                        )

            ExplainService.save_as_file(filename="../input/inputImage.jpg",content=input_image_content)

        except Exception as e:
            log.error(e,exc_info=True)
            raise Exception

        # local_explain_image_model = joblib.load("../models/local_explain_image_model.pkl")
        inputImage = cv2.imread('../input/inputImage.jpg')
        obj_explain = ResponsibleAIExplain.local_image_explain(inputImage=inputImage
                                                               ,model=local_explain_image_model
                                                               ,method= payload.methods
                                                               ,segmentation_fn=payload.segmentationType
                                                               )
        log.debug(f"obj_explain: {obj_explain}")

        method= payload.methods
        if(method=="INTEGRATED-GRADIENTS"):
            norm_attr = visualization._normalize_image_attr(obj_explain['attributions'], sign='all')
            default_cmap = LinearSegmentedColormap.from_list("RdWhGn", ["red", "white", "green"])

            plt.imshow(np.mean(inputImage, axis=2), cmap="gray") 
            plt.imshow(norm_attr, cmap=default_cmap, vmin=-1, vmax=1, alpha=0.5)
            plt.axis('off')
            plt.savefig("../output/outputImageIG.jpg", bbox_inches='tight', pad_inches=0)

            with open("../output/outputImageIG.jpg", "rb") as imagefile:
                convert_outputImage = base64.b64encode(imagefile.read())
           
            objExplainabilityLocalResponse =ExplainabilityLocalImageResponse(
                explanation=convert_outputImage,
                segments=""
            )
            local_file_path = "../output/localImageIG_explain.json"
            ExplainService.save_as_json_file(local_file_path,objExplainabilityLocalResponse.dict())

            bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(payload.outputPath)
            NutanixObjectStorage.upload_with_high_threshold(
                bucket_name=bucket_dict['bucket_name'],
                object_key=bucket_dict['object_key'],
                file_size_mb=10,
                local_file_path=local_file_path
            )

            object_key = "/".join(bucket_dict['object_key'].split("/")[:-1])
            log.debug(f"object_key: {object_key}")
            
            NutanixObjectStorage.upload_with_high_threshold(
                bucket_name=bucket_dict['bucket_name'],
                object_key=f"{object_key}/outputImageIG.jpg",
                file_size_mb=10,
                local_file_path="../output/outputImageIG.jpg"
            )
            log.debug("Created Output Image")
            log.info("returning:")
            
            return objExplainabilityLocalResponse
        elif(method=="LIME"):
            plt.imshow(obj_explain['lime_image'])
            plt.axis('off')
            plt.savefig("../output/outputImageLime.jpg", bbox_inches='tight', pad_inches=0)
            with open("../output/outputImageLime.jpg", "rb") as imagefile:
                convert_outputImage = base64.b64encode(imagefile.read())

            objExplainabilityLocalResponse = ExplainabilityLocalImageResponse(
                explanation=convert_outputImage,
                segments=""
            )
            local_file_path = "../output/localImageLime_explain.json"
            ExplainService.save_as_json_file(local_file_path,objExplainabilityLocalResponse.dict())

            bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(payload.outputPath)
            NutanixObjectStorage.upload_with_high_threshold(
                bucket_name=bucket_dict['bucket_name'],
                object_key=bucket_dict['object_key'],
                file_size_mb=10,
                local_file_path=local_file_path
            )

            object_key = "/".join(bucket_dict['object_key'].split("/")[:-1])
            log.debug(f"object_key: {object_key}")
            
            NutanixObjectStorage.upload_with_high_threshold(
                bucket_name=bucket_dict['bucket_name'],
                object_key=f"{object_key}/outputImageLime.jpg",
                file_size_mb=10,
                local_file_path="../output/outputImageLime.jpg"
            )
            return objExplainabilityLocalResponse

        else:
            plt.imshow(obj_explain['anchor'])
            plt.axis('off')
            plt.savefig("../output/outputImage.jpg", bbox_inches='tight', pad_inches=0)

            plt.imshow(obj_explain['segments'])
            plt.axis('off')
            plt.savefig("../output/outputSegments.jpg", bbox_inches='tight', pad_inches=0)

            with open("../output/outputImage.jpg", "rb") as imagefile:
                convert_outputImage = base64.b64encode(imagefile.read())

            with open("../output/outputSegments.jpg", "rb") as imagefile:
                convert_outputSegments = base64.b64encode(imagefile.read())

            objExplainabilityLocalResponse = ExplainabilityLocalImageResponse(
                explanation=convert_outputImage,
                segments=convert_outputSegments
            )
            local_file_path = "../output/localImage_explain.json"
            ExplainService.save_as_json_file(local_file_path,objExplainabilityLocalResponse.dict())

            bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(payload.outputPath)
            NutanixObjectStorage.upload_with_high_threshold(
                bucket_name=bucket_dict['bucket_name'],
                object_key=bucket_dict['object_key'],
                file_size_mb=10,
                local_file_path=local_file_path
            )

            object_key = "/".join(bucket_dict['object_key'].split("/")[:-1])
            log.debug(f"object_key: {object_key}")
            
            NutanixObjectStorage.upload_with_high_threshold(
                bucket_name=bucket_dict['bucket_name'],
                object_key=f"{object_key}/outputImage.jpg",
                file_size_mb=10,
                local_file_path="../output/outputImage.jpg"
            )
            log.debug("Created Output Image")
            NutanixObjectStorage.upload_with_high_threshold(
                bucket_name=bucket_dict['bucket_name'],
                object_key=f"{object_key}/outputSegments.jpg",
                file_size_mb=10,
                local_file_path="../output/outputSegments.jpg"
            )
            log.debug("Created Output Segment")
            
            return objExplainabilityLocalResponse

    def local_explain_demo (payload: ExplainabilityLocalDemoRequest) -> ExplainabilityLocalResponseDemo:
        log.debug(f"payload: {payload}")

        try:
            obj_explain = ResponsibleAIExplain.local_explain_demo(text=payload.inputText
                                                            ,class_names=["Negative","Positive"]
                                                            )
            log.debug(f"obj_explain: {obj_explain}")
            
            List_explain = []
            List_explain.append(obj_explain)

            objExplainabilityLocalResponse = ExplainabilityLocalResponseDemo(
                explainerID=payload.explainerID,
                explanation=List_explain
            )

            return objExplainabilityLocalResponse
        except Exception as e:
            log.error(f"UUID: {request_id_var.get()}, Error: {e}", exc_info=True)
            Tbl_Exception.create({"UUID":request_id_var.get(),"function":"local_explain_demoServiceFunction","msg":str(e),"description":str(e)+"Line No:"+str(e.__traceback__.tb_lineno)})
            raise 
            # Tbl_Exception.create({"UUID":request_id_var.get(),"function":"local_explain_demoServiceFunction","msg":str(e),"description":str(e)+"Line No:"+str(e.__traceback__.tb_lineno)})
            # return ExplainabilityLocalResponseDemo(explainerID=payload.explainerID, explanation=[])

    def localTabular_explain (payload: ExplainabilityLocalTabularRequest) -> ExplainabilityLocalTabularResponse:
        log.debug(f"payload: {payload}")

        try:
            bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(payload.modelPredictionEndpoint)

            start_time = datetime.now()
            model_file_content = NutanixObjectStorage.get_file_content(bucket_dict["bucket_name"]
                                                  ,bucket_dict["object_key"])
            end_time = datetime.now()
            log.debug(f"total time for downloading file: {end_time - start_time}")
            ExplainService.save_as_file(filename="../models/local_explain_classifier.pkl",content=model_file_content)

            if payload.preprocessorPredictionEndpoint is not None:
                bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(payload.preprocessorPredictionEndpoint)
                model_file_content = NutanixObjectStorage.get_file_content(bucket_dict["bucket_name"],
                                                                           bucket_dict["object_key"])

                ExplainService.save_as_file(filename="../models/local_explain_preprocessor.pkl", content=model_file_content)

            bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(payload.datasetPath)
            data = NutanixObjectStorage.get_file_content(bucket_dict["bucket_name"],
                                                         bucket_dict["object_key"])

            dataset = None
            if payload.datasetType=='text/csv':
                data = str(data, 'utf-8')
                dataset = pd.read_csv(StringIO(data))
            elif payload.datasetType=='text/parquet':
                dataset = pd.read_parquet(BytesIO(data),engine='pyarrow')

        except Exception as e:
            log.error(e,exc_info=True)
            raise Exception

        local_explain_classifier = joblib.load("../models/local_explain_classifier.pkl")
        if payload.preprocessorPredictionEndpoint is not None:
            local_explain_preprocessor = joblib.load("../models/local_explain_preprocessor.pkl")
        else:
            local_explain_preprocessor = None

        obj_explain = ResponsibleAIExplain.local_explain_tabular(dataset=dataset,
                                                                 inputIndex=payload.inputIndex,
                                                                 taskType=payload.taskType,
                                                                 method=payload.methods,
                                                                 model=local_explain_classifier,
                                                                 preprocessor=local_explain_preprocessor,
                                                                 featureNames=payload.featureNames,
                                                                 categoricalFeatureNames=payload.categoricalFeatureNames,
                                                                 targetNames=payload.targetNames,
                                                                 targetClassNames=payload.targetClassNames
                                                                 )
        log.debug(f"obj_explain: {obj_explain}")
        
        List_explain_tabular = []
        for item in obj_explain:
            log.debug(f"item: {item}")
            
            predictedTarget = item['predictedTarget']
            if item.get('anchor'):
                objexplainabilitylocalabular = ExplainabilityLocalTabular(modelPrediction=predictedTarget,
                                                                            modelConfidence=item['modelConfidence'],
                                                                            anchor=item['anchor'],
                                                                            importantFeatures = None,
                                                                            limeTimeSeries=None,
                                                                            inputRow=item['inputRow'])
            elif item.get('importantFeatures'):
                objexplainabilitylocalabular = ExplainabilityLocalTabular(modelPrediction=predictedTarget,
                                                                            modelConfidence=item['modelConfidence'],
                                                                            anchor=None,
                                                                            importantFeatures=item['importantFeatures'],
                                                                            limeTimeSeries=None,
                                                                            inputRow=item['inputRow'])
            elif item.get('limeTimeSeries'):
                item['limeTimeSeries'][0].savefig("../output/LimeTimeseries.jpg")
                with open("../output/LimeTimeseries.jpg", "rb") as imagefile:
                    convert_outputImage = base64.b64encode(imagefile.read())

                objexplainabilitylocalabular = ExplainabilityLocalTabular(modelPrediction=predictedTarget,
                                                                            modelConfidence=item['modelConfidence'],
                                                                            anchor=None,
                                                                            importantFeatures=None,
                                                                            limeTimeSeries=convert_outputImage,
                                                                            inputRow=None)
            else:
                objexplainabilitylocalabular = ExplainabilityLocalTabular(predictedTarget=predictedTarget,inputRow=[])
            
            List_explain_tabular.append(objexplainabilitylocalabular)
        
        objExplainabilityLocalResponse = ExplainabilityLocalTabularResponse(
            explainerID=payload.explainerID,
            explanation=List_explain_tabular
        )
        
        local_file_path = "../output/localTabular_explain.json"
        ExplainService.save_as_json_file(local_file_path,objExplainabilityLocalResponse.dict())

        bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(payload.outputPath)

        NutanixObjectStorage.upload_with_high_threshold(
            bucket_name=bucket_dict['bucket_name'],
            object_key=bucket_dict['object_key'],
            file_size_mb=10,
            local_file_path=local_file_path
        )

        return objExplainabilityLocalResponse

    def global_explain(payload: ExplainabilityGlobalRequest) -> ExplainabilityGlobalResponse:
        log.debug(f"payload: {payload}")

        try:
            bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(payload.modelPredictionEndpoint)
            model_file_content = NutanixObjectStorage.get_file_content(bucket_dict["bucket_name"]
                                                  ,bucket_dict["object_key"])

            ExplainService.save_as_file(filename="../models/global_explain_model.pkl",content=model_file_content)

            if payload.preprocessorPredictionEndpoint is not None:
                bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(payload.preprocessorPredictionEndpoint)

                model_file_content = NutanixObjectStorage.get_file_content(bucket_dict["bucket_name"]
                                                                           ,bucket_dict["object_key"])

                try:
                    ExplainService.save_as_file(filename="../models/global_explain_processor.pkl", content=model_file_content)
                
                except Exception as e:
                    log.info(f"Processor pkl not found")

            bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(payload.datasetPath)
            data = NutanixObjectStorage.get_file_content(bucket_dict["bucket_name"]
                                                                       ,bucket_dict["object_key"])
            dataset = None
            if payload.datasetType == 'text/csv':
                data = str(data, 'utf-8')
                dataset = pd.read_csv(StringIO(data))
            elif payload.datasetType == 'text/parquet':
                dataset = pd.read_parquet(BytesIO(data), engine='pyarrow')

        except Exception as e:
            log.error(e,exc_info=True)
            raise Exception

        global_explain_model = joblib.load("../models/global_explain_model.pkl")

        global_explain_processor = None
        if payload.preprocessorPredictionEndpoint is not None:
            try:
                global_explain_processor = joblib.load("../models/global_explain_processor.pkl")
            except Exception as e:
                log.info(f"Handled Exception: {e}")

        result_global_explain = ResponsibleAIExplain.global_explain(dataset=dataset
                                                          ,taskType= payload.taskType
                                                          ,method= payload.methods
                                                          ,model=global_explain_model
                                                          ,preprocessor=global_explain_processor
                                                          ,featureNames=payload.featureNames
                                                          ,categoricalFeatureNames=payload.categoricalFeatureNames
                                                          ,targetNames=payload.targetNames
                                                          ,targetClassNames = payload.targetClassNames
                                                        )
        log.debug(f"result_global_explain: {result_global_explain}")
        log.debug(f"result_global_explain: {type(result_global_explain)}")

        list_explaination =[]
        for item in result_global_explain:
            log.error(f"item: {item}")
            
            list_importantFeatures = []
            for imp_feature in item["importantFeatures"]:
                log.error(f"imp_feature: {imp_feature}")
                obj_importantFeatures = importantFeatures(name=imp_feature['featureName']
                                                          ,value=imp_feature['importanceScore'])
                list_importantFeatures.append(obj_importantFeatures)

            obj_explainabilityglobal = explainabilityglobal(importantFeatures=list_importantFeatures)

            list_explaination.append(obj_explainabilityglobal)

        objExplainabilityResponse = ExplainabilityGlobalResponse(
            explanation = list_explaination
        )
        log.debug(f"objExplainabilityResponse: {objExplainabilityResponse}")
        
        local_file_path = "../output/global_explain.json"
        ExplainService.save_as_json_file(local_file_path,objExplainabilityResponse.dict())

        bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(payload.outputPath)
        NutanixObjectStorage.upload_with_high_threshold(
            bucket_name=bucket_dict['bucket_name'],
            object_key=bucket_dict['object_key'],
            file_size_mb=10,
            local_file_path=local_file_path
        )

        return objExplainabilityResponse
    
    def process_file_input(payload: dict):

        try:
            datasetFile = payload['datasetFile']
            modelFile = payload['modelFile']
            vectorizerFile = payload['vectorizerFile']
            datasetColumns = None
            unique_id = uuid4().hex

            os.makedirs(f'../uploads/{unique_id}')
            if datasetFile is not None :

                with open(f'../uploads/{unique_id}/{datasetFile.filename}','wb') as file:
                    file.write(datasetFile.file.read())

                log.info(f"datasetFile: {datasetFile.filename}")
                log.info(f"datasetFile: {datasetFile.content_type}")

                NutanixObjectStorage.upload_with_high_threshold(local_file_path=f'../uploads/{unique_id}/{datasetFile.filename}',
                                                                bucket_name="responsible-ai-demo",
                                                                object_key=f"rai-explain/{unique_id}/data.csv",
                                                                file_size_mb=50)
                if datasetFile.content_type == 'text/csv':
                    df = pd.read_csv(f'../uploads/{unique_id}/{datasetFile.filename}')
                elif datasetFile.content_type =='text/parquet':
                    df = pd.read_parquet(f'../uploads/{unique_id}/{datasetFile.filename}')
                else:
                    log.error("Invalid File Format")
                    log.error(f"File Format : {datasetFile.filename}")
                    raise Exception

                datasetColumns = list(df.columns)

            with open(f'../uploads/{unique_id}/{modelFile.filename}','wb') as file:
                file.write(modelFile.file.read())

            log.info(f"modelFile: {modelFile.filename}")
            log.info(f"modelFile: {modelFile.content_type}")
            if modelFile.filename.split(".")[-1] == "pkl":
                NutanixObjectStorage.upload_with_high_threshold(local_file_path=f'../uploads/{unique_id}/{modelFile.filename}',
                                                                bucket_name="responsible-ai-demo",
                                                                object_key=f"rai-explain/{unique_id}/model.pkl",
                                                                file_size_mb=50)
            elif modelFile.filename.split(".")[-1] == "h5":
                NutanixObjectStorage.upload_with_high_threshold(local_file_path=f'../uploads/{unique_id}/{modelFile.filename}',
                                                                bucket_name="responsible-ai-demo",
                                                                object_key=f"rai-explain/{unique_id}/model.h5",
                                                                file_size_mb=50)
            else:
                log.error("Invalid File Format")
                log.error(f"File Format : {modelFile.filename}")
                raise Exception
            if vectorizerFile is not None:
                with open(f'../uploads/{unique_id}/{vectorizerFile.filename}','wb') as file:
                    file.write(vectorizerFile.file.read())

                NutanixObjectStorage.upload_with_high_threshold(local_file_path=f'../uploads/{unique_id}/{vectorizerFile.filename}',
                                                                bucket_name="responsible-ai-demo",
                                                                object_key=f"rai-explain/{unique_id}/vectorizer.pkl",
                                                                file_size_mb=50)
                
                datasetColumns = None

            obj = ExplainFileOutput(status="SUCCESS",
                                    datasetColumns=datasetColumns,
                                    uniqueId=unique_id)

        except Exception as e:
            log.error(e,exc_info=True)
        finally:
            try:
                shutil.rmtree(f'../uploads/{unique_id}/')
            except Exception as e:
                log.error(e,exc_info=True)
                log.info("Excepion Handled")
        return obj

    def process_payload_input(payload: dict):

        try:
            scope = payload.scope
            dataType = payload.dataType
            method = payload.method
            uniqueId = payload.uniqueId
            inputText = payload.inputText
            response_payload = read_config_yaml('../config/response_payload.yaml')[scope][dataType][method]
            try:
                url = os.getenv("ADMIN_URL")
                exp_ip_key = os.getenv("EXP_IP_KEY")
                response = requests.request("GET", url, verify=False).json()
                result = response['result']
                api_config = {
                    'GLOBAL': {'STRUCTURE': result['Exp_Global_Tabular']},
                    'LOCAL': {
                        'STRUCTURE': result['Exp_Local_Tabular'],
                        'UNSTRUCTURE': result['Exp_Local_Text'],
                        'IMAGE': result['Exp_Local_Image']
                    }
                }
                rai_explain_api = result[exp_ip_key] + api_config[scope][dataType]
            except Exception as e:
                log.error(e,exc_true=True)
                obj = ExplainOutput(status="FAILURE", payload={}, api="")
                return obj
            bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(response_payload['modelPredictionEndpoint']+uniqueId)
            bucket = NutanixObjectStorage.s3.Bucket(bucket_dict['bucket_name'])
            
            for obj in bucket.objects.filter(Prefix=bucket_dict["object_key"]):
                if 'model' in obj.key:
                    modelFileType =obj.key.split(".")[-1]
                if 'vectorizer' in obj.key:
                    vectorizerFileType =obj.key.split(".")[-1]
                if 'data' in obj.key:
                    dataFileType =obj.key.split(".")[-1]
            response_payload['modelPredictionEndpoint'] = response_payload['modelPredictionEndpoint']+uniqueId+"//model."+modelFileType
            response_payload['taskType'] = payload.taskType
            if payload.targetClassNames is not None and payload.targetClassNames !=[]:
                response_payload['targetClassNames'] = payload.targetClassNames
            response_payload['outputPath'] = response_payload['outputPath']+uniqueId+"//output.json"
            if dataType == 'UNSTRUCTURE' or dataType == 'TEXT':
                response_payload['inputText'] = inputText
                bucket_dict = NutanixObjectStorage.parse_nutanix_bucket_object(response_payload['vectorizerPredictionEndpoint']+uniqueId)
                response_payload['vectorizerPredictionEndpoint'] = response_payload['vectorizerPredictionEndpoint']+uniqueId+"//vectorizer."+vectorizerFileType
            elif dataType == 'TABULAR' or dataType == 'STRUCTURE':
                response_payload['datasetPath'] = response_payload['datasetPath']+uniqueId+"//data."+dataFileType
                response_payload['targetNames'] = payload.targetNames
                if response_payload.get('inputIndex'):
                    response_payload['inputIndex'] = randint(0,50)
            else:
                log.error("Invalid DataType")
                raise Exception
            obj = ExplainOutput(status="SUCCESS",payload=response_payload,api=rai_explain_api)
            
            return obj
        except Exception as e:
            log.error(e, exc_info=True)
            obj = ExplainOutput(status="FAILURE", payload={}, api="")
            return obj
    
    def get_explanation_methods(payload: dict):
        # Check if payload is not None and it contains 'modelId' and 'datasetId'
        if payload.modelId is None or '' or payload.datasetId is None or '':
            log.error("modelId and/or datasetId are missing")
            return GetExplanationMethodsResponse(status='FAILURE', 
                                                 message='modelId and/or datasetId are missing', 
                                                 dataType='',
                                                 methods=[])

        try:
            # Extract modelId and datasetId from the payload
            modelId = payload.modelId
            datasetId = payload.datasetId

            model_attribute_ids = ModelAttributes.find(model_attributes=['useModelApi'])
            model_attribute_values = ModelAttributeValues.find(batch_id=None, model_id=modelId, model_attribute_ids=model_attribute_ids)
            
            use_model_end_point = model_attribute_values[0]
            if use_model_end_point.lower() == 'yes':
                model_attribute_names = ['modelFramework','taskType']
                model_attribute_ids = ModelAttributes.find(model_attributes=model_attribute_names)
                model_attribute_values = ModelAttributeValues.find(batch_id=None, model_id=modelId, model_attribute_ids=model_attribute_ids)

                model_details = {'modelFramework': model_attribute_values[0],
                                 'taskType': model_attribute_values[1]}
            else:
                model_attribute_names = ['modelFramework', 'algorithm', 'taskType']
                model_attribute_ids = ModelAttributes.find(model_attributes=model_attribute_names)
                model_attribute_values = ModelAttributeValues.find(batch_id=None, model_id=modelId, model_attribute_ids=model_attribute_ids)

                model_details = {'modelFramework': model_attribute_values[0],
                                'algorithm': model_attribute_values[1],
                                'taskType': model_attribute_values[2]}
            
            dataset_attribute_ids = DatasetAttributes.find(dataset_attributes=['dataType'])
            dataType = DatasetAttributeValues.find(dataset_id=datasetId, dataset_attribute_ids=dataset_attribute_ids)
            
            dataset_details = {'dataType': dataType[0]}
            
            # Get the explanation methods for the given modelType, taskType, and dataType
            cursor = Tbl_Explanation_Methods.find_methods(model_framework=model_details['modelFramework'], 
                                                         task_type=model_details['taskType'], 
                                                         data_type=dataset_details['dataType'])
        
            # Check if the cursor is not None
            if not cursor:
                log.error("No explanation methods found")
                return GetExplanationMethodsResponse(status='FAILURE', message='No explanation methods found',dataType='', methods=[])
            
            method_list = []
            if payload.scope is not None:
                scope = payload.scope
                if use_model_end_point.lower() == 'yes':
                    for document in cursor:
                        # Check the scope
                        if document['scope'] == scope:
                            method_list.append(document['methods'])
                else:
                    # Create a list of explanation methods for the given scope
                    for document in cursor:
                        # Check if the modelType is not in the unsupportedModelTypes list for the given explanation method
                        if document['scope'] == scope and model_details['algorithm'].split('(')[0] not in document['unsupportedModels']:
                            method_list.append(document['methods'])
            else:
                if use_model_end_point.lower() == 'yes':
                    for document in cursor:
                        if document['methods'] not in method_list:
                                method_list.append(document['methods'])
                else:
                    # Create a list of explanation methods for LOCAL and GLOBAL scopes
                    for document in cursor:
                        # Check if the modelType is not in the unsupportedModelTypes list for the given explanation method
                        if model_details['algorithm'].split('(')[0] not in document['unsupportedModels']:
                            if document['methods'] not in method_list:
                                method_list.append(document['methods'])
            
            # check if method_list is empty or not, if empty raise exception
            if not method_list:
                raise ValueError("No explanation methods found for the given modelType, taskType, and dataType")
            
            # Create a GetExplanationMethodsResponse object with the scope and methods and return it
            obj = GetExplanationMethodsResponse(status='SUCCESS', 
                                                message='Identification of explanation methods successful',
                                                dataType=dataset_details['dataType'],
                                                methods=method_list
                                                )
            return obj

        except Exception as e:
            # Log the error and return an empty GetExplanationMethodsResponse object
            log.error(f"UUID: {request_id_var.get()}, Error: {e}", exc_info=True)
            Tbl_Exception.create({"UUID":request_id_var.get(),"function":"get_explanation_methodsServiceFunction","msg":str(e),"description":str(e)+"Line No:"+str(e.__traceback__.tb_lineno)})
            raise
            # return GetExplanationMethodsResponse(status='FAILURE', 
                                                #  message=str(e), 
                                                #  dataType='',
                                                #  methods=[])

    
    def data_to_dataframe(data, column_name='Value'):
        """
        Convert a single string or a dictionary to a pandas DataFrame.
        
        For a single string, create a single-column DataFrame.
        For a dictionary, create a single-row DataFrame with each key-value pair as a column-value pair.
        
        Parameters:
        data (str or dict): The data to be converted into a DataFrame.
        column_name (str): The name of the DataFrame column if the data is a string. Default is 'Value'.
        
        Returns:
        pd.DataFrame: The resulting DataFrame with the data.
        """
        if isinstance(data, str):
            # Create a single-column DataFrame for a single string
            df = pd.DataFrame([data], columns=[column_name])
        elif isinstance(data, dict):
            # Create a single-row DataFrame with each key-value pair as a column-value pair for a dictionary
            df = pd.DataFrame([data])
        else:
            raise ValueError("The input data must be either a single string or a dictionary")

        return df
    
    def generate_explanation(payload: dict):
        try:
            db_type = os.getenv('DB_TYPE').lower()

            # Extract modelId and datasetId from the payload
            modelId = payload.modelId
            datasetId = payload.datasetId
            scope = payload.scope
            method = payload.method 
            if hasattr(payload, 'inputText'):
                inputText = payload.inputText
            else:
                inputText = None

            if hasattr(payload, 'inputRow'):
                inputRow = payload.inputRow
            else:
                inputRow = None
            preprocessorId = payload.preprocessorId

            model_attribute_ids = ModelAttributes.find(model_attributes=['useModelApi'])
            model_attribute_values = ModelAttributeValues.find(batch_id=None, model_id=modelId, model_attribute_ids=model_attribute_ids)
            use_model_end_point = model_attribute_values[0]

            model_details = Model.find(model_id=modelId)

            if use_model_end_point.lower() == 'yes':
                model_attribute_names = ['modelFramework','taskType','data','prediction','targetDataType']
                model_attribute_ids = ModelAttributes.find(model_attributes=model_attribute_names)
                model_attribute_values = ModelAttributeValues.find(batch_id=None, model_id=modelId, model_attribute_ids=model_attribute_ids)
               
                model_details['modelFramework'] = model_attribute_values[0]
                model_details['taskType'] = model_attribute_values[1]
                model_details['data'] = model_attribute_values[2]
                model_details['prediction'] = model_attribute_values[3]
                model_details['targetDataType'] = model_attribute_values[4]
                model_details['algorithm'] = None

            else:
                # Get the model details
                model_attribute_ids = ModelAttributes.find(model_attributes=['modelFramework', 'algorithm', 'taskType','targetDataType'])
                model_attribute_values = ModelAttributeValues.find(batch_id=None, model_id=modelId, model_attribute_ids=model_attribute_ids)
                model_details['modelFramework'] = model_attribute_values[0]
                model_details['algorithm'] = model_attribute_values[1]
                model_details['taskType'] = model_attribute_values[2]
                model_details['targetDataType'] = model_attribute_values[3]
                model_details['data'] = None
                model_details['prediction'] = None
                model_details['ModelEndPoint'] = None
            
            # Load the model based on its Access/Framework type
            if model_details['modelFramework'] == 'API':
                model = model_details['ModelEndPoint']
            else:
                container_name = None if db_type == 'mongo' else os.getenv('MODEL_CONTAINER_NAME')
                modelObject = fileStoreDb.read_file_exp(unique_id=model_details['ModelData'], container_name=container_name)

                if model_details['modelFramework'] in ('Scikit-learn', 'Statsmodels'):
                    model = joblib.load(BytesIO(modelObject['data'].read()) if db_type == 'mongo' else BytesIO(modelObject['data']))
                elif model_details['modelFramework'] == 'Keras':
                    with open('model.h5', 'wb') as f:
                        f.write(modelObject['data'].read() if db_type == 'mongo' else modelObject['data'])
                        model = keras.models.load_model('model.h5')
                    os.remove('model.h5')
                else:
                    log.error("Unsupported model file type. Supported file types are pkl/h5")
                    return GetExplanationResponse(status='FAILURE', message='Unsupported model file type. Supported file types are pkl/h5', explanation=[])

            # Get the dataset details 
            if model_details['taskType'] == 'CLASSIFICATION' or model_details['taskType'] == 'CLUSTERING':
                dataset_attributes = ['groundTruthClassLabel', 'dataType', 'fileName', 'groundTruthClassNames']
            else:
                dataset_attributes = ['groundTruthClassLabel', 'dataType', 'fileName']
            dataset_details = Dataset.find(dataset_id=datasetId)
            dataset_attribute_ids = DatasetAttributes.find(dataset_attributes=dataset_attributes)
            dataset_attribute_values = DatasetAttributeValues.find(dataset_id=datasetId, dataset_attribute_ids=dataset_attribute_ids)

            dataset_details['targetClassLabel'] = dataset_attribute_values[0]
            dataset_details['dataType'] = dataset_attribute_values[1]
            dataset_details['fileName'] = dataset_attribute_values[2]
            dataset_details['datasetFileType'] = dataset_attribute_values[2].split('.')[-1]
            try:
                dataset_details['targetClassNames'] = dataset_attribute_values[3]
            except IndexError:
                dataset_details['targetClassNames'] = None
            
            # Load the dataset based on its file type
            container_name = None if db_type == 'mongo' else os.getenv('DATASET_CONTAINER_NAME')
            datasetObject = fileStoreDb.read_file_exp(unique_id = dataset_details['SampleData'], container_name = container_name)
            data = BytesIO(datasetObject['data'].read()) if db_type == 'mongo' else BytesIO(datasetObject['data'])
            if dataset_details['datasetFileType'] == 'csv':
                try:
                    dataset = pd.read_csv(data)
                except UnicodeDecodeError:
                    data.seek(0)  # Reset the read position of the BytesIO object
                    try:
                        dataset = pd.read_csv(data, encoding='ISO-8859-1')
                    except UnicodeDecodeError:
                        dataset = pd.read_csv(data, encoding='cp1252')
            elif dataset_details['datasetFileType'] == 'parquet':
                dataset = pd.read_parquet(data)
            else:
                log.error("Unsupported dataset file type")
                return GetExplanationResponse(status='FAILURE', message='Unsupported dataset file type. Supported file types are csv/parquet.', explanation=[])

            if inputRow is not None:
                lineDataset_as_is = ExplainService.data_to_dataframe(inputRow)
                
                # Get the common columns between datasetA and datasetB
                common_columns = [col for col in dataset.columns if col in lineDataset_as_is.columns]

                # Reorder columns in datasetB to match the order in datasetA
                lineDataset = lineDataset_as_is[common_columns]
                
            elif inputText is not None:
                lineDataset = ExplainService.data_to_dataframe(inputText)
            else:
                lineDataset = None
            
            if preprocessorId is not None and preprocessorId != 0:
                preprocessor_details = Preprocessor.find(preprocessor_id= preprocessorId)

                container_name = None if db_type == 'mongo' else os.getenv('PREPROCESSOR_CONTAINER_NAME')
                preprocessorObject = fileStoreDb.read_file_exp(unique_id=preprocessor_details['PreprocessorFileId'], container_name=container_name)
                preprocessor = joblib.load(BytesIO(preprocessorObject['data'].read()) if db_type == 'mongo' else BytesIO(preprocessorObject['data']))
            else:
                preprocessor = None
            
            inputIndex = 0
            # Generate the explanation based on the method
            obj_explain = ResponsibleAIExplain.get_explanation(model=model,
                                                                taskType=model_details['taskType'],
                                                                modelType=model_details['modelFramework'],
                                                                dataset=dataset,
                                                                preprocessor=preprocessor,
                                                                targetClassLabel=dataset_details['targetClassLabel'],
                                                                targetClassNames=dataset_details['targetClassNames'],
                                                                method=method,
                                                                scope=scope,
                                                                lineDataset=lineDataset,
                                                                inputIndex=inputIndex,
                                                                api_input_request= model_details['data'],
                                                                api_output_response= model_details['prediction']
                                                                )
    
            List_explain_tabular = []
            model_dict={"Model Name":model_details['ModelName'],"Algorithm":model_details['algorithm'],"ModelEndpoint": model_details['ModelEndPoint'],"Task Type": model_details['taskType'][0] + model_details['taskType'][1:].lower()}
            dataset_dict={"Dataset Name":dataset_details['DataSetName'], "dataType": dataset_details['dataType'], "GroundTruth Class Label":dataset_details['targetClassLabel'],"GroundTruth Class Names": dataset_details['targetClassNames']}
            for item in obj_explain: 
                if item.get('anchor'):
                    objexplainabilitylocalabular = ExplainabilityTabular_New(modelName = model_dict['Model Name'],
                                                                            algorithm = model_dict['Algorithm'],
                                                                            endpoint = model_dict['ModelEndpoint'],
                                                                            taskType = model_dict['Task Type'],
                                                                            datasetName = dataset_dict['Dataset Name'],
                                                                            dataType = dataset_dict['dataType'],
                                                                            groundTruthLabel = dataset_dict['GroundTruth Class Label'], 
                                                                            groundTruthClassNames = dataset_dict['GroundTruth Class Names'],
                                                                            featureNames = item['featureNames'],
                                                                            methodName = 'ANCHOR',
                                                                            methodDescription = item['description'],
                                                                            anchor=item['anchor'],
                                                                            attributionsText =None,
                                                                            featureImportance = None,
                                                                            timeSeriesForecast=None,
                                                                            shapValues=None,
                                                                            explanationDesc=None
                                                                                )
                elif item.get('attributionsText'):
                    objexplainabilitylocalabular = ExplainabilityTabular_New(modelName = model_dict['Model Name'],
                                                                                algorithm = model_dict['Algorithm'],
                                                                                endpoint = model_dict['ModelEndpoint'],
                                                                                taskType = model_dict['Task Type'],
                                                                                datasetName = dataset_dict['Dataset Name'],
                                                                                dataType = dataset_dict['dataType'],
                                                                                groundTruthLabel = dataset_dict['GroundTruth Class Label'], 
                                                                                groundTruthClassNames = dataset_dict['GroundTruth Class Names'],
                                                                                featureNames = item['featureNames'],
                                                                                methodName = 'INTEGRATED GRADIENTS',
                                                                                methodDescription = item['description'],
                                                                                anchor = None,
                                                                                attributionsText = item['attributionsText'],
                                                                                featureImportance = None,
                                                                                timeSeriesForecast = None,
                                                                                shapValues = None,
                                                                                explanationDesc = None
                                                                                )
                elif item.get('importantFeatures'):
                    method_mapping = {
                        'KERNEL-SHAP': 'KERNEL SHAP',
                        'TREE-SHAP': 'TREE SHAP',
                        'PERMUTATION-IMPORTANCE': 'PERMUTATION IMPORTANCE',
                        'PD-VARIANCE': 'PARTIAL DEPENDENCE VARIANCE',
                        'LIME-TABULAR': 'LIME TABULAR'
                    }
                    method_name = method_mapping.get(method)
                    objexplainabilitylocalabular = ExplainabilityTabular_New(modelName = model_dict['Model Name'],
                                                                            algorithm = model_dict['Algorithm'],
                                                                            endpoint = model_dict['ModelEndpoint'],
                                                                            taskType = model_dict['Task Type'],
                                                                            datasetName = dataset_dict['Dataset Name'],
                                                                            dataType = dataset_dict['dataType'],
                                                                            groundTruthLabel = dataset_dict['GroundTruth Class Label'], 
                                                                            groundTruthClassNames = dataset_dict['GroundTruth Class Names'],
                                                                            featureNames = item['featureNames'],
                                                                            methodName = method_name,
                                                                            methodDescription = item['description'],
                                                                            anchor=None,
                                                                            attributionsText =None,
                                                                            featureImportance = item['importantFeatures'],
                                                                            timeSeriesForecast=None,
                                                                            shapValues=None,
                                                                            explanationDesc=None)
                elif item.get('timeSeries'):
                    method_mapping = {
                        'TS-LIMEEXPLAINER': 'LIME EXPLAINER',
                        'TS-LIME-TABULAR-EXPLAINER': 'LIME TABULAR EXPLAINER',
                        'TS-SHAPEXPLAINER': 'SHAP EXPLAINER'
                    }
                    method_name = method_mapping.get(method)
                    item['timeSeries'][0]['timeSeries'][0].savefig("../output/LimeTimeseries.jpg")
                    with open("../output/LimeTimeseries.jpg", "rb") as imagefile:
                        convert_outputImage = base64.b64encode(imagefile.read()).decode('utf-8')
                    item['timeSeries'][0]['timeSeries'] = convert_outputImage
                    objexplainabilitylocalabular = ExplainabilityTabular_New(modelName = model_dict['Model Name'],
                                                                            algorithm = model_dict['Algorithm'],
                                                                            endpoint = model_dict['ModelEndpoint'],
                                                                            taskType = model_dict['Task Type'],
                                                                            datasetName = dataset_dict['Dataset Name'],
                                                                            dataType = dataset_dict['dataType'],
                                                                            groundTruthLabel = dataset_dict['GroundTruth Class Label'], 
                                                                            groundTruthClassNames = dataset_dict['GroundTruth Class Names'],
                                                                            featureNames = item['featureNames'],
                                                                            methodName = method_name,
                                                                            methodDescription = item['description'],
                                                                            anchor = None,
                                                                            attributionsText = None,
                                                                            featureImportance = None,
                                                                            timeSeriesForecast = item['timeSeries'],
                                                                            shapValues = None,
                                                                            explanationDesc = None)
                else:
                    objexplainabilitylocalabular = ExplainabilityTabular_New(predictedTarget=item['predictedTarget'],inputRow=[])
                    
                List_explain_tabular.append(objexplainabilitylocalabular.dict())
                
            return GetExplanationResponse(status='SUCCESS',
                                                message='Explanation generated successfully',
                                                explanation=List_explain_tabular)
            
        except Exception as e:
            log.error(f"UUID: {request_id_var.get()}, Error: {e}", exc_info=True)
            Tbl_Exception.create({"UUID":request_id_var.get(),"function":"generate_explanationServiceFunction","msg":str(e),"description":str(e)+"Line No:"+str(e.__traceback__.tb_lineno)})
            raise
    
    def generate_report(payload: dict):
        # Check if payload is not None and it contains 'modelId' and 'datasetId'
        if payload.batchId is None or '':
            log.error("Batch Id id missing")
            return GetReportResponse(status='FAILURE', 
                                     message='Batch Id missing')
        try:
            tenet_id = Tenet.find(tenet_name='Explainability')
            batch_id = payload.batchId
            batch_details = Batch.find(batch_id=batch_id, tenet_id=tenet_id)
            modelId = batch_details['ModelId']
            datasetId = batch_details['DataId']
            preprocessorId = batch_details['PreprocessorId']
            title =  batch_details['Title']

            model_attribute_ids = ModelAttributes.find(model_attributes=['taskType'])
            model_attribute_values = ModelAttributeValues.find(batch_id=None, model_id=modelId, model_attribute_ids=model_attribute_ids)                
            task_type = model_attribute_values[0]

            if task_type != 'TIMESERIESFORECAST': 
                updated_methods = {"GLOBAL":["KERNEL-SHAP"], "LOCAL": ["LIME-TABULAR"]}
            else:
                model_attribute_ids = ModelAttributes.find(model_attributes=['appExplanationMethods'])
                model_attribute_values = ModelAttributeValues.find(batch_id=batch_id, model_id=modelId, model_attribute_ids=model_attribute_ids)    
                methods = model_attribute_values[0]
                
                updated_methods = {"LOCAL": methods}


            # Update the batch status to "Started"
            Batch.update(batch_id=batch_id, value={'Status': "Started"})
            
            final_response=[]
            
            for scope, values in updated_methods.items():
                for method in values:
                    response={}
                    
                    # Create a Payload object
                    payload_obj = Payload(modelId=modelId, datasetId=datasetId, preprocessorId=preprocessorId, 
                                          scope=scope, method=method)
                   
                    # Generate explanation for the given payload
                    obj_explain = ExplainService.generate_explanation(payload_obj)
                    
                    response["title"] = title
                    response["algorithm"] = obj_explain.explanation[0].algorithm
                    response["endpoint"] = obj_explain.explanation[0].endpoint
                    response["taskType"] = obj_explain.explanation[0].taskType
                    response["datasetName"] = obj_explain.explanation[0].datasetName
                    response["dataType"] = obj_explain.explanation[0].dataType
                    response["groundTruthLabel"] = obj_explain.explanation[0].groundTruthLabel
                    response["groundTruthClassNames"] = obj_explain.explanation[0].groundTruthClassNames
                    response["methodName"] = obj_explain.explanation[0].methodName
                    response["methodDescription"] = obj_explain.explanation[0].methodDescription
                    response["scope"] = scope
                    response["featureNames"] = obj_explain.explanation[0].featureNames
                   
                    # Check if 'anchor' is in obj_explain
                    if obj_explain.explanation[0].anchor is not None:
                        response["anchors"] = obj_explain.explanation[0].anchor
                    # Check if 'attributionsText' is in obj_explain
                    elif obj_explain.explanation[0].attributionsText is not None:
                        response["attributionsText"] = obj_explain.explanation[0].attributionsText
                    # Check if 'featureImportance' is in obj_explain
                    elif obj_explain.explanation[0].featureImportance is not None:
                        response["featureImportance"]=obj_explain.explanation[0].featureImportance
                    # Check if 'limeTimeSeries' is in obj_explain
                    elif obj_explain.explanation[0].timeSeriesForecast is not None:
                        response["timeSeriesForecast"] = obj_explain.explanation[0].timeSeriesForecast
                    # If none of the above, set 'anchors', 'featureImportance', 'attributionsText', and 'limeTimeSeries' to None
                    else:
                        response["anchors"] = None
                        response["attributionsText"] = None
                        response["featureImportance"]=None
                        response["limeTimeSeries"]=None

                    final_response.append(response)
            
            # Generate HTML content from the final_response
            html_content = Report.generate_html_content(final_response)
            
            # Save the HTML content to a local file
            local_file_path = "../output/explanationreport.html"
            ExplainService.save_html_to_file(html_content, local_file_path)

            if obj_explain.explanation[0].taskType in ["Classification", "Regression"]:
                # Create csv file
                CreateCSV.json_to_csv(final_response)

                # Define the directory containing the files to be zipped and the zip file path
                output_dir = '../output'
                zip_file_path = os.path.join(output_dir, 'report.zip')

                # Ensure the output directory exists
                os.makedirs(output_dir, exist_ok=True)

                # Create a zip file and add all .csv and .html files in the output directory to it
                with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            if file.endswith('.csv') or file.endswith('.html'):
                                file_path = os.path.join(root, file)
                                # Adjust the path in the zip file to maintain the directory structure under 'output/'
                                arcname = os.path.relpath(file_path, os.path.dirname(output_dir))
                                zipf.write(file_path, arcname)

                # Now you can save the zip file to the database
                # Assuming fileStoreDb.save_file() takes a file-like object, you can open the zip file in binary mode
                with open('../output/report.zip', 'rb') as zipf:
                    FileId = fileStoreDb.save_file(file = zipf, filename = 'explanation_report.zip', contentType = 'application/zip', tenet = 'Explainability')
                    report_name = 'explanation_report.zip'
            
            elif obj_explain.explanation[0].taskType in ["Timeseriesforecast"]:
                FileId = fileStoreDb.save_file(file = BytesIO(html_content.encode('utf-8')), filename = 'explainability_report.html',contentType = 'text/html', tenet='Explainability')
                report_name = 'explainability_report.html'

            HtmlId = time.time()
            doc = {
                    'HtmlId': HtmlId,
                    'BatchId': batch_id,
                    'TenetId': tenet_id,
                    'ReportName': report_name,
                    'HtmlFileId': FileId,
                    'CreatedDateTime': datetime.now()
                }
            Html.create(doc)

            url = os.getenv("REPORT_URL")
            payload = {"batchId": batch_id}
            response = requests.request("POST", url, data=payload, verify=False).json()

            # Directory path of the output folder
            output_dir = '../output'
            # Delete all files in the directory
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            if response['status']=='SUCCESS':
                # Update the batch status to "Completed"
                Batch.update(batch_id=batch_id, value={'Status': "Completed"})
                return GetReportResponse(status='SUCCESS',
                                        message='Report generated successfully')
            else:
                # Update the batch status to "Failed"
                Batch.update(batch_id=batch_id, value={'Status': "Failed"})
                return GetReportResponse(status='FAILURE',
                                        message=f"Error in generating report due to: {response['message']}")
            
        except Exception as e:
            Batch.update(batch_id=batch_id, value={'Status': "Failed"})
            log.error(f"UUID: {request_id_var.get()}, Error: {e}", exc_info=True)
            Tbl_Exception.create({"UUID":request_id_var.get(),"function":"generate_reportServiceFunction","msg":str(e),"description":str(e)+"Line No:"+str(e.__traceback__.tb_lineno)})
            raise
            
    def mask_pdf(payload: dict):
        try:
            # Check if payload is not None and it contains a valid PDF file
            if 'pdfFile' not in payload or payload['pdfFile'].filename == '':
                log.error("PDF file is missing")
                raise MissingPDFException()
            
            os.makedirs(f'../uploads/')
            unmasked_images_folder = f'../uploads/unmasked_images_folder'
            os.makedirs(unmasked_images_folder)
            masked_images_folder = f'../uploads/masked_images_folder'
            os.makedirs(masked_images_folder)

            # Save the PDF file to a local directory
            pdf_file_path = f"../uploads/{payload['pdfFile'].filename}"
            with open(pdf_file_path, 'wb') as file:
                file.write(payload['pdfFile'].file.read())

            folders = os.listdir(unmasked_images_folder)
            if len(folders)>0:
                for folder in os.listdir(unmasked_images_folder):
                    os.remove(os.path.join(unmasked_images_folder, folder))
            
            pdf_to_img(pdf_file_path, unmasked_images_folder)

            for image_filename in os.listdir(unmasked_images_folder):
                image_path = os.path.join(unmasked_images_folder, image_filename)
                with open(image_path, 'rb') as image_file:
                    # Prepare the files dictionary to send the image as a file upload
                    files = {'image': (image_filename, image_file, 'image/png')}
                    privacy_payload = {"ocr": "ComputerVision",
                            "magnification": "False",
                            "rotationFlag": "False",
                            "portfolio": "usecase",
                            "account": "Healthcare"
                            }

                    # Send the request with the files parameter for file upload
                    response = requests.request("POST", os.getenv("PRIVACY_IMAGE_ANONYMIZE"), data=privacy_payload, files=files).json()
                    masked_image = base64.b64decode(response)
                    masked_image_path = os.path.join(masked_images_folder, image_filename)
                    with open(masked_image_path, 'wb') as masked_image_file:
                        masked_image_file.write(masked_image)

            # Convert the masked images to a PDF file
            images_to_pdf(masked_images_folder, f"../uploads/processed_{payload['pdfFile'].filename}")
            
            # Read processed pdf file
            with open(f"../uploads/processed_{payload['pdfFile'].filename}", 'rb') as file:
                processed_pdf = file.read()

            response = StreamingResponse(BytesIO(processed_pdf), media_type='application/pdf; charset=utf-8')
            response.headers["Content-Disposition"] = 'attachment; filename='+f"processed_{payload['pdfFile'].filename}"
        except Exception as e:
            log.error(f"UUID: {request_id_var.get()}, Error: {e}", exc_info=True)
            Tbl_Exception.create({"UUID":request_id_var.get(),"function":"generate_reportServiceFunction","msg":str(e),"description":str(e)+"Line No:"+str(e.__traceback__.tb_lineno)})
            raise
        finally:
            try:
                shutil.rmtree(f'../uploads/')
            except Exception as e:
                log.error(f"UUID: {request_id_var.get()}, Error: {e}",exc_info=True)
                raise
        return response
    
    def moderation_cot(payload: dict):
        try:
            # Check if payload is not None and it contains all the required fields
            if payload.Prompt is None or payload.Response is None:
                log.error("Prompt/Response are missing in payload")
                raise ValueError("Prompt/Response are missing in the payload")
            
            payload.Prompt = payload.Prompt.replace('Provide justification for your answer.', 'Please consider the given question and reference then let us know in simple terms how the answer is derived.')
            payload.Prompt = payload.Prompt + ' ' + payload.Response

            # Prepare the payload for the moderation API
            moderation_payload = {
                "Prompt": payload.Prompt,
                "temperature": "0",
                "model_name": "gpt4"
            }

            llm_explanation_payload = {
                "inputPrompt": payload.Prompt,
                "response": payload.Response
            }

            # Prepare headers for the request
            headers = {
                "Content-Type": "application/json"
            }

            try:
                # Send the request to the moderation API & LLM Explanation API
                cot_response = requests.request("POST", os.getenv("MODERATION_API"), json=moderation_payload, headers=headers).json()
            except:
                cot_response = {}
            try:
                llm_explanation_response = requests.request("POST", os.getenv("LLM_EXPLANATION_API"), json=llm_explanation_payload, headers=headers).json()
            except:
                llm_explanation_response = {}
            
            # Return the response from the moderation API
            return ExplainabilityCoTResponse(CoT = cot_response, Explanation = llm_explanation_response)

        except Exception as e:
            log.error(f"UUID: {request_id_var.get()}, Error: {e}", exc_info=True)
            Tbl_Exception.create({"UUID":request_id_var.get(),"function":"generate_reportServiceFunction","msg":str(e),"description":str(e)+"Line No:"+str(e.__traceback__.tb_lineno)})
            raise