"""
Copyright 2024 Infosys Ltd.”

Use of this source code is governed by MIT license that can be found in the LICENSE file or at
MIT license https://opensource.org/licenses/MIT

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from typing import Annotated, Any, Dict, List

from fairness.config.logger import CustomLogger
from fairness.exception.exception import FairnessException
from fairness.mappers.mappers import BiasAnalyzeRequest, BiasAnalyzeResponse, GetMitigationRequest, IndividualFairnessRequest, MitigateBiasRequest, GetBiasResponse, \
    GetBiasRequest, MitigationAnalyzeResponse, PreprocessingMitigationAnalyzeResponse, PreprocessingMitigateBiasRequest,BatchId
from fairness.service.service import FairnessService
from fairness.service.service import FairnessUIservice
from fairness.service.service_latest import FairnessServiceUpload
from fairness.service.service_latest import FairnessUIserviceUpload
from fairness.service.preprocessing import FairnessServicePreproc
from fairness.service.preprocessing import FairnessUIservicePreproc
from fairness.service.postprocessing import ModelMitigation
from fairness.service.bert import BertService
from fastapi import APIRouter, HTTPException, UploadFile, Form,Body
from requests.exceptions import ChunkedEncodingError
# from starlette.responses import JSONResponse
from fastapi.responses import StreamingResponse,FileResponse, Response
from fairness.mappers.mappers import BiasPretrainMitigationResponse
from fairness.dao.WorkBench.FileStoreDb import FileStoreReportDb
from fairness.exception.custom_exception import CustomHTTPException
from fairness.Telemetry.Telemetry_call import SERVICE_UPLOAD_FILE_METADATA
from fairness.Telemetry.Telemetry_call import SERVICE_UPD_GETATTRIBUTE_METADATA
from fairness.Telemetry.Telemetry_call import SERVICE_attributes_Data_METADATA
from fairness.Telemetry.Telemetry_call import SERVICE_return_protected_attrib_DB_METADATA
from fairness.Telemetry.Telemetry_call import SERVICE_getLabels_METADATA
from fairness.Telemetry.Telemetry_call import SERVICE_getScore_METADATA
from fairness.Telemetry.Telemetry_call import SERVICE_upload_file_Premitigation_METADATA
from fairness.Telemetry.Telemetry_call import SERVICE_return_pretrainMitigation_protected_attrib_METADATA
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from fairness.auth.auth_client_id import get_auth_client_id
from fairness.auth.auth_jwt import get_auth_jwt
from fairness.auth.auth_none import get_auth_none
import traceback
import io
import uuid 
import os
import requests
import concurrent.futures
from typing import Optional

router = APIRouter()
AnalyseUpload = APIRouter()
PretrainMitigateUpload = APIRouter()
IndividualMetricUpload = APIRouter()
Analyze = APIRouter()
Preprocessing_mitigation = APIRouter()
model_mitigation_router = APIRouter()
individual_metrics_router = APIRouter()
fine_tuned_bert_router=APIRouter()
log = CustomLogger()
uiService = FairnessUIservice()
service = FairnessService()
analyse = FairnessServiceUpload()
analyseUpload = FairnessUIserviceUpload()
servicepreproc = FairnessServicePreproc()
uiServicepreproc = FairnessUIservicePreproc()
modelMitigate = ModelMitigation()
fairnesstelemetryurl = os.getenv("FAIRNESS_TELEMETRY_URL")
telemetry_flag = os.getenv("tel_Falg")
from fairness.Telemetry.Telemetry_call import SERVICE_attributes_Data_METADATA

auth_type = os.environ.get("AUTH_TYPE")

if auth_type == "azure":
    auth = get_auth_client_id()
elif auth_type == "jwt":
    auth = get_auth_jwt()

elif auth_type == 'none':
    auth = get_auth_none()
else:
    raise HTTPException(status_code=500, detail="Invalid authentication type configured")
    

@router.post('/fairness/bias/analyze', response_model=BiasAnalyzeResponse)
def analyze(payload: BiasAnalyzeRequest, auth= Depends(auth)):
    log.info("Entered create usecase routing method")
    try:
        log.info("before invoking create usecase service ")
        log.debug("request payload: " + str(payload))
        response = service.analyze(payload)
        log.info("after invoking create usecase service ")
        log.debug("response : " + str(response))
        log.info("exit create usecase routing method")
        return response
    except Exception as cie:
            log.error(cie.__dict__)
            log.info("exit create usecase routing method")
            raise Exception(str(cie))


@router.post('/fairness/bias/mitigate', response_model=MitigationAnalyzeResponse)
def migrate(payload: MitigateBiasRequest, auth= Depends(auth)):
    try:
        log.info("before invoking create usecase mitigation service ")
        log.debug("request payload: " + str(payload))
        log.debug(str(payload))
        response = service.mitigate(payload)
        log.debug(response)
        return response
    except Exception as cie:
            log.error(cie.__dict__)
            log.info("exit create usecase routing method")
            raise Exception(str(cie.__dict__))

@router.post('/fairness/bias/mitigate/preprocessing/mitigatedDataset', response_model=BiasPretrainMitigationResponse)
def preprocessingmigrate(payload: PreprocessingMitigateBiasRequest, auth= Depends(auth)):
    try:
        log.info("before invoking create usecase mitigation service ")
        log.debug("request payload: " + str(payload))
        log.debug(str(payload))
        response = service.preprocessingmitigate(payload)
        log.debug(response)
        return response
    except Exception as cie:
            log.error(cie.__dict__)
            log.info("exit create usecase routing method")
            raise Exception(str(cie.__dict__))

@router.get('/fairness/bias/get/{mlModelId}', response_model=GetBiasResponse)
def process(payload: GetBiasRequest, auth= Depends(auth)):
    try:
        log.debug(str(payload))
        response = service.analyze(payload)
        log.debug(response)
        return response
    except FairnessException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)

@router.get('/fairness/mitigation/getMitigatedData/{fileName}')
def process(fileName: str, auth= Depends(auth)):
    try:
        log.debug(fileName)
        content_type=""
        if fileName.endswith(".csv"):
            content_type="text/csv"
        elif fileName.endswith(".feather"):
            content_type="application/octet-stream"
        elif fileName.endswith(".parquet"):
            content_type="application/octet-stream"
        elif fileName.endswith(".json"):
             content_type="application/json" 
        headers = {
            'Content-Disposition': f'attachment; filename={fileName}',
            "Content-Type": content_type
        }
        response = service.get_mitigated_data(fileName)
        return FileResponse(response, headers=headers)

    except FairnessException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)

@router.post('/fairness/bias/Workbench/fileid')
def get_data(biasType: Annotated[str, Form()],
             methodType: Annotated[str, Form()],
             taskType: Annotated[str, Form()] ,
             fileId: Annotated[str, Form()], auth= Depends(auth)):

    
    payload = {"biasType": biasType,
               "methodType": methodType,
               "taskType": taskType,
               "fileId": fileId}
    try:
        log.info("before invoking create usecase service ")
        log.debug("request payload: " + str(payload))
        log.debug(str(payload))
        response = uiService.attributes_Data(payload)
        log.debug(response)
        return response
    except HTTPException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        tb = traceback.TracebackException.from_exception(cie)
        last_traceback = list(tb.format())[-2]  
        errormsg = str(last_traceback)
        raise CustomHTTPException(SERVICE_attributes_Data_METADATA, "error occured"+ " Line no:" + str(errormsg), cie.__dict__.get("detail"))
    except ChunkedEncodingError as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        tb = traceback.TracebackException.from_exception(cie)
        last_traceback = list(tb.format())[-2]  
        errormsg = str(last_traceback)
        raise CustomHTTPException(SERVICE_attributes_Data_METADATA, "error occured"+ " Line no:" + str(errormsg), "file not found")


@router.post('/fairness/bias/UIworkbench/batchId/Attributes')
def analyze(payload: BatchId, auth= Depends(auth)):
    log.info("Entered create usecase routing method")
    try:
        log.info("before invoking create usecase service ")
        log.debug("request payload: " + str(payload))
        response = uiService.return_protected_attrib_DB(payload)
        log.info("after invoking create usecase service ")
        log.debug("response : " + str(response))
        log.info("exit create usecase routing method")
        return response
    except HTTPException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        tb = traceback.TracebackException.from_exception(cie)
        last_traceback = list(tb.format())[-2]  
        errormsg = str(last_traceback)
        raise CustomHTTPException(SERVICE_return_protected_attrib_DB_METADATA, "error occured"+ " Line no:" + str(errormsg),cie.__dict__.get("detail"))
    except ChunkedEncodingError as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        tb = traceback.TracebackException.from_exception(cie)
        last_traceback = list(tb.format())[-2]  
        errormsg = str(last_traceback)
        raise CustomHTTPException(SERVICE_return_protected_attrib_DB_METADATA, "error occured"+ " Line no:" + str(errormsg), "file not found")

   
@router.post('/fairness/bias/getDataset')
def get_data(biasType: Annotated[str, Form()],
             methodType: Annotated[str, Form()],
             taskType: Annotated[str, Form()],
             fileId: Annotated[str, Form()], auth= Depends(auth)):
    payload = {"biasType": biasType,
               "methodType": methodType,
               "taskType": taskType,
               "fileId": fileId}
    try:
        log.info("before invoking create usecase service ")
        log.debug("request payload: " + str(payload))
        log.debug(str(payload))
        response = uiService.upload_file(payload)
        log.debug(response)
        return response
    except HTTPException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        tb = traceback.TracebackException.from_exception(cie)
        last_traceback = list(tb.format())[-2] 
        errormsg = str(last_traceback)
        raise CustomHTTPException(SERVICE_UPLOAD_FILE_METADATA, "error occured"+ " Line no:" + str(errormsg), cie.__dict__.get("detail"))
    except ChunkedEncodingError as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        tb = traceback.TracebackException.from_exception(cie)
        last_traceback = list(tb.format())[-2]  
        errormsg = str(last_traceback)
        raise CustomHTTPException(SERVICE_UPLOAD_FILE_METADATA, "error occured"+ " Line no:" + str(errormsg), "file not found")



@router.post('/fairness/bias/getAttributes')
def get_attributes(Label: Annotated[str, Form()],
                   FavourableOutcome: Annotated[str, Form()],
                   ProtectedAttribute: Annotated[str, Form()],
                   Priviledged: Annotated[str, Form()],
                   recordId: Annotated[str,Form()], auth= Depends(auth)):
    payload = {"label": Label,
               "FavourableOutcome": FavourableOutcome,
               "ProtectedAttribute": ProtectedAttribute,
               "priviledged": Priviledged,
               "recordId": recordId}
    try:
        log.info("before invoking create usecase service ")
        log.debug("request payload: " + str(payload))
        log.debug(str(payload))
        response = uiService.return_protected_attrib(payload)
        log.debug(response)
        return response
    except HTTPException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        tb = traceback.TracebackException.from_exception(cie)
        last_traceback = list(tb.format())[-2]  
        errormsg = str(last_traceback)
        raise CustomHTTPException(SERVICE_UPD_GETATTRIBUTE_METADATA, "error occured"+ " Line no:" + str(errormsg), cie.__dict__.get("detail"))
    except ChunkedEncodingError as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        tb = traceback.TracebackException.from_exception(cie)
        last_traceback = list(tb.format())[-2]  
        errormsg = str(last_traceback)
        raise CustomHTTPException(SERVICE_UPD_GETATTRIBUTE_METADATA, "error occured"+ " Line no:" + str(errormsg), "file not found")
    

    

@router.post('/fairness/mitigation/getDataset')
def get_data(biasType: Annotated[str, Form()],
             methodType: Annotated[str, Form()],
             taskType: Annotated[str, Form()],
             trainingDataset: UploadFile,
             predictionDataset:UploadFile, auth= Depends(auth)
             ):
    payload = {"biasType": biasType,
               "methodType": methodType,
               "taskType": taskType,
               "trainFile": trainingDataset,
               "testFile":predictionDataset
               }
    try:
        log.info("before invoking create usecase getdata mitigation service ")
        log.debug("request payload: " + str(payload))
        log.debug(str(payload))
        response = uiService.upload_file_mitigation(payload)
        log.debug(response)
        return response
    except FairnessException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase getdata mitigation routing method")
        raise HTTPException(**cie.__dict__)   
    
 
@router.post('/fairness/mitigation/getAttributes')
def get_attributes(MitigationType: Annotated[str,Form()],
                   MitigationTechnique: Annotated[str,Form()],
                   Label: Annotated[str, Form()],
                   FavourableOutcome: Annotated[str, Form()],
                   ProtectedAttribute: Annotated[str, Form()],
                   Priviledged: Annotated[str, Form()], auth= Depends(auth)):
    payload = {
                "mitigationType":MitigationType,
                "mitigationTechnique":MitigationTechnique,
                "label": Label,
                "FavourableOutcome": FavourableOutcome,
                "ProtectedAttribute": ProtectedAttribute,
                "priviledged": Priviledged}
    try:
        log.info("before invoking create get attribute mitigation usecase service ")
        log.debug("request payload: " + str(payload))
        log.debug(str(payload))
        response = uiService.return_protected_attrib_mitigation(payload)
        log.debug(response)
        return response
    except FairnessException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase get attribute mitigation routing method")
        raise HTTPException(**cie.__dict__)


@router.post('/fairness/pretrain/mitigation/getDataset')
def get_data(MitigationType: Annotated[str, Form()],
             MitigationTechnique: Annotated[str, Form()],
             taskType: Annotated[str, Form()],
             fileId: Annotated[str, Form()], auth= Depends(auth)
             ):
    payload = {"mitigationType": MitigationType,
               "mitigationTechnique": MitigationTechnique,
               "taskType": taskType,
               "fileId": fileId
               }
    try:
        log.info("before invoking create usecase getdata mitigation service ")
        log.debug("request payload: " + str(payload))
        log.debug(str(payload))
        response = uiService.upload_file_pretrainMitigation(payload)
        log.debug(response)
        return response
    except HTTPException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        tb = traceback.TracebackException.from_exception(cie)
        last_traceback = list(tb.format())[-2]  
        errormsg = str(last_traceback)
        raise CustomHTTPException(SERVICE_upload_file_Premitigation_METADATA, "error occured"+ " Line no:" + str(errormsg), cie.__dict__.get("detail"))
    except ChunkedEncodingError as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        tb = traceback.TracebackException.from_exception(cie)
        last_traceback = list(tb.format())[-2]  
        errormsg = str(last_traceback)
        raise CustomHTTPException(SERVICE_upload_file_Premitigation_METADATA, "error occured"+ " Line no:" + str(errormsg), "file not found")

    
 
@router.post('/fairness/pretrain/mitigation/getAttributes')
def get_attributes(
                   Label: Annotated[str, Form()],
                   FavourableOutcome: Annotated[str, Form()],
                   ProtectedAttribute: Annotated[str, Form()],
                   Priviledged: Annotated[str, Form()],
                   recordId: Annotated[str, Form()], auth= Depends(auth)):
    payload = {
                "label": Label,
                "FavourableOutcome": FavourableOutcome,
                "ProtectedAttribute": ProtectedAttribute,
                "priviledged": Priviledged,
                "recordId": recordId}
    try:
        log.info("before invoking create get attribute mitigation usecase service ")
        log.debug("request payload: " + str(payload))
        log.debug(str(payload))
        response = uiService.return_pretrainMitigation_protected_attrib(payload)
        log.debug(response)
        return response
    except HTTPException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        tb = traceback.TracebackException.from_exception(cie)
        last_traceback = list(tb.format())[-2]  
        errormsg = str(last_traceback)
        raise CustomHTTPException(SERVICE_return_pretrainMitigation_protected_attrib_METADATA, "error occured"+ " Line no:" + str(errormsg), cie.__dict__.get("detail"))
    except ChunkedEncodingError as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        tb = traceback.TracebackException.from_exception(cie)
        last_traceback = list(tb.format())[-2]  
        errormsg = str(last_traceback)
        raise CustomHTTPException(SERVICE_return_pretrainMitigation_protected_attrib_METADATA, "error occured"+ " Line no:" + str(errormsg), "file not found")




@router.post('/fairness/mitigation/model/uploadFiles')
def mitigation_model_upload_files(
    model: UploadFile,
    trainingDataset: UploadFile,
    testingDataset: UploadFile, auth= Depends(auth)
): 
    payload={
        'model': model,
        'trainingDataset': trainingDataset,
        'testingDataset': testingDataset
    }

    try:
        log.info('before invoking mitigation model upload files service')

        log.debug('request payload: '+str(payload))
        
        response=uiService.mitigation_model_upload_files(payload)

        log.debug('response: '+str(response))

        return response
    except FairnessException as cie:
        log.error(cie.__dict__)

        log.info('exit mitigation model upload files service')

        raise HTTPException(**cie.__dict__)

@router.post('/fairness/mitigation/model/getMitigatedModelNameAnalyze')
def mitigation_model_get_mitigated_model_name_analyze(
    recordId: Annotated[str, Form()],
    label: Annotated[str, Form()],
    sensitiveFeatures: Annotated[str, Form()], auth= Depends(auth)
):

    payload={
        'recordId': recordId,
        'label': label,
        'sensitiveFeatures': sensitiveFeatures
    }

    try:
        log.info('before invoking mitigation model get mitigated model name analyze service')

        log.debug('request payload: '+str(payload))

        response=uiService.mitigation_model_get_mitigated_model_name_analyze(payload)

        log.debug('response: '+str(response))

        return response
    except FairnessException as cie:
        log.error(cie.__dict__)

        log.info('exit mitigation model get mitigated model name service')

        raise HTTPException(**cie.__dict__)

@router.get('/fairness/mitigation/model/getMitigatedModel/{filename}')
def get_mitigated_model(filename: str, auth= Depends(auth)):
    try:
        log.info('before invoking mitigation model get mitigated model service')

        log.debug(filename)

        response=uiService.get_mitigated_model(filename)

        log.debug(response)

        return FileResponse(
            response,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except FairnessException as cie:
        log.error(cie.__dict__)

        log.info('exit mitigation model get mitigated model service')

        raise HTTPException(**cie.__dict__)

@router.post('/fairness/inprocessing/exponentiated_gradient_reduction')
def inprocessing_exponentiated_gradient_reduction(
    trainingDataset: UploadFile,
    testingDataset: UploadFile,
    label: Annotated[str, Form()],
    favourableOutcome: Annotated[str, Form()],
    sensitiveFeatures: Annotated[str, Form()], auth= Depends(auth)
):
    try:
        log.info('before invoking inprocessing exponentiated gradient reduction service')

        payload={
            'trainingDataset': trainingDataset,
            'testingDataset': testingDataset,
            'label': label,
            'favourableLabel': favourableOutcome,
            'sensitiveFeatures': sensitiveFeatures
        }

        log.debug('request payload: '+str(payload))

        response=uiService.inprocessing_exponentiated_gradient_reduction(payload)

        log.debug('response: '+str(response))

        return response
    except FairnessException as cie:
        log.error(cie.__dict__)

        log.info('exit inprocessing exponentiated gradient reduction service')

        raise HTTPException(**cie.__dict__)

@router.get('/fairness/inprocessing/getModel/{filename}')
def inprocessing_get_model(filename: str, auth= Depends(auth)):
    try:
        log.info('before invoking inprocessing get model service')

        log.debug(filename)

        response=uiService.inprocessing_get_model(filename)

        log.debug(response)

        return FileResponse(
            response,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except FairnessException as cie:
        log.error(cie.__dict__)

        log.info('exit inprocessing get model service')

        raise HTTPException(**cie.__dict__)    

@router.post('/fairness/analysis/llm')
def get_analysis_llm(
    response: Annotated[str, Form()], 
    evaluator: Annotated[str, Form()], auth= Depends(auth)
):
    try:
        payload={
            'response': response,
            'evaluator': evaluator
        }

        log.info('before invoking get analysis llm service')

        log.debug('request payload: '+str(payload))

        response=uiService.get_analysis_llm(payload)

        log.debug('response: '+str(response))

        return response
    except FairnessException as cie:
        log.error(cie.__dict__)

        log.info('exit get analysis llm service')

        raise HTTPException(**cie.__dict__)
    
@router.post('/fairness/analysis/llm/insert_analysis_details')
def insert_analysis_details(category: str, name: str, value: str, active: bool, addedBy: str, auth= Depends(auth)):
    try:
        payload={
            'category': category,
            'name': name,
            'value': value,
            'active': active,
            'addedBy': addedBy
        }
        
        log.info('before invoking insert analysis details service')

        log.debug('request payload: '+str(payload))

        response=uiService.insert_analysis_details(payload)

        log.debug('response: '+str(response))

        return response
    except FairnessException as cie:
        log.error(cie.__dict__)

        log.info('exit insert analysis details service')

        raise HTTPException(**cie.__dict__)

@router.post('/fairness/analysis/llm/insert_connection_credentials')
def insert_connection_credentials(name: str, value: str, details: dict, active: bool, auth= Depends(auth)):
    try: 
        payload={
            'name': name,
            'value': value,
            'details': details,
            'active': active
        }

        log.info('before invoking insert connection credentials service')

        log.debug('request payload: '+str(payload))

        response=uiService.insert_connection_credentials(payload)

        log.debug('response: '+str(response))

        return response
    except FairnessException as cie:
        log.error(cie.__dict__)

        log.info('exit insert connection credentials service')

        raise HTTPException(**cie.__dict__)

@router.get('/fairness/analysis/llm/get_analysis_details')
def get_analysis_details(auth= Depends(auth)):
    try: 
        log.info('before invoking get analysis details service')

        response=uiService.get_analysis_details()

        log.debug('response: '+str(response))

        return response
    except FairnessException as cie:
        log.error(cie.__dict__)

        log.info('exit get analysis details service')

        raise HTTPException(**cie.__dict__)

@router.get('/fairness/analysis/llm/get_connection_credentials')
def get_connection_credentials(auth= Depends(auth)):
    try: 
        log.info('before invoking get connection credentials service')

        response=uiService.get_connection_credentials()

        log.debug('response: '+str(response))

        return response
    except FairnessException as cie:
        log.error(cie.__dict__)

        log.info('exit get connection credentials service')

        raise HTTPException(**cie.__dict__)

#upload file to local db, if one does not have schema of workbench
#uncomment both router and service method if required
# @router.post('/fairness/local/workbench/uploadFile')
# def local_uploadFile(
#     file: UploadFile
# ): 
#     payload={
#         'file': file
#     }
#     try:        
#         response=uiService.local_uploadFile(payload)
#         return response
#     except FairnessException as cie:
#         raise HTTPException(**cie.__dict__)
    
    
    
#Get individual metrics scores
@individual_metrics_router.post('/fairness/individual/bias/getlabels')
def get_labels(fileId: Annotated[str, Form()], auth= Depends(auth)):
    payload = {"fileId": fileId}
    try:
        log.info("before invoking create usecase service ")
        log.debug("request payload: " + str(payload))
        log.debug(str(payload))
        response = uiService.getLabels(payload)
        log.debug(response)
        return response
    except HTTPException as cie:
        log.error(cie.__dict__)
        print("inside exception data")
        log.info("exit create usecase routing method")
        traceback.print_exc()
        tb = traceback.TracebackException.from_exception(cie)
        last_traceback = list(tb.format())[-2]  
        print(last_traceback, "last_line")
        errormsg = str(last_traceback)
        raise CustomHTTPException(SERVICE_getLabels_METADATA, "filenot found error"+ " Line no:" + str(errormsg),  cie.__dict__.get("detail"))
    except ChunkedEncodingError as cie:
        log.error(cie.__dict__)
        print("inside exception data")
        log.info("exit create usecase routing method")
        traceback.print_exc()
        tb = traceback.TracebackException.from_exception(cie)
        last_traceback = list(tb.format())[-2]  
        print(last_traceback, "last_line")
        errormsg = str(last_traceback)
        raise CustomHTTPException(SERVICE_getLabels_METADATA, "error occured"+ " Line no:" + str(errormsg), "file not found")



@individual_metrics_router.post('/fairness/individual/bias/getscore',response_model=List[Dict[str, Any]])
def get_individualScore(payload:IndividualFairnessRequest, auth= Depends(auth)):
    
    try:
        log.info("before invoking create usecase service ")
        log.debug("request payload: " + str(payload))
        log.debug(str(payload))
        response = uiService.getScore(payload)
        log.debug(response)
        return response
    except HTTPException as cie:
        log.error(cie.__dict__)
        print("inside exception data")
        log.info("exit create usecase routing method")
        traceback.print_exc()
        tb = traceback.TracebackException.from_exception(cie)
        last_traceback = list(tb.format())[-2] 
        print(last_traceback, "last_line")
        errormsg = str(last_traceback)
        raise CustomHTTPException(SERVICE_getScore_METADATA, "filenot found error"+ " Line no:" + str(errormsg),  cie.__dict__.get("detail"))
    except ChunkedEncodingError as cie:
        log.error(cie.__dict__)
        print("inside exception data")
        log.info("exit create usecase routing method")
        traceback.print_exc()
        tb = traceback.TracebackException.from_exception(cie)
        last_traceback = list(tb.format())[-2]  
        print(last_traceback, "last_line")
        errormsg = str(last_traceback)
        raise CustomHTTPException(SERVICE_getScore_METADATA, "error occured"+ " Line no:" + str(errormsg), "file not found")



@router.post('/fairness/analysis/image')
def get_analysis_image(
    prompt: Annotated[str, Form()],
    image: UploadFile,
    evaluator: Annotated[str, Form()], auth= Depends(auth)
):
    try:
        payload={
            'prompt': prompt,
            'image': image,
            'evaluator': evaluator
        }

        log.info('before invoking get analysis image service')

        log.debug('request payload: '+str(payload))
        
        response=uiService.get_analysis_image(payload)

        log.debug('response: '+str(response))

        return response
    except FairnessException as cie:
        log.error(cie.__dict__)

        log.info('exit get analysis image service')

        raise HTTPException(**cie.__dict__)


@AnalyseUpload.post('/fairness/Analyse')
def Analyse_UploadFile(
                    biasType: Annotated[str, Form()],
                    methodType: Annotated[str, Form()],
                    taskType: Annotated[str, Form()],
                    Label: Annotated[str, Form()],
                    predLabel:Annotated[str,Form()],
                    FavourableOutcome: Annotated[str, Form()],
                    ProtectedAttribute: Annotated[str, Form()],
                    Priviledged: Annotated[str, Form()],
                    file: UploadFile, auth= Depends(auth)):
    payload = {
                "biasType": biasType,
                "methodType": methodType,
                "taskType": taskType,
                "label": Label,
                "predLabel":predLabel,
                "FavourableOutcome": FavourableOutcome,
                "ProtectedAttribute": ProtectedAttribute,
                "priviledged": Priviledged, 
                "file": file}
    
    log.info("before invoking create get attribute mitigation usecase service ")
    log.debug("request payload: " + str(payload))
    log.debug(str(payload))
    response = analyseUpload.upload_file(payload)
    log.debug(response)
    return response


@PretrainMitigateUpload.post('/fairness/pretrainMitigate')
def Mitigate_UploadFile( 
                    MitigationType: Annotated[str, Form()],
                    MitigationTechnique: Annotated[str, Form()],
                    taskType: Annotated[str, Form()],
                    Label: Annotated[str, Form()],
                    FavourableOutcome: Annotated[str, Form()],
                    ProtectedAttribute: Annotated[str, Form()],
                    Priviledged: Annotated[str, Form()],
                    file: UploadFile, auth= Depends(auth)):
    payload = {
                "MitigationType": MitigationType,
                "MitigationTechnique": MitigationTechnique,
                "taskType": taskType,
                "label": Label,
                "FavourableOutcome": FavourableOutcome,
                "ProtectedAttribute": ProtectedAttribute,
                "priviledged": Priviledged, 
                "file": file}
    log.info("before invoking create get attribute mitigation usecase service ")
    log.debug("request payload: " + str(payload))
    log.debug(str(payload))
    response = analyseUpload.upload_file_Premitigation(payload)
    log.debug(response)
    return response


@IndividualMetricUpload.post('/fairness/IndividualMetrics',response_model=List[Dict[str, Any]])
def Individual_UploadFile( 
                    label : List[str],
                    file: UploadFile, auth= Depends(auth)):
    payload = {
                "label": label, 
                "file": file}
    log.info("before invoking create get attribute mitigation usecase service ")
    log.debug("request payload: " + str(payload))
    log.debug(str(payload))
    response = analyseUpload.getLabels_Individual(payload)
    log.debug(response)
    return response
    
    
@fine_tuned_bert_router.post('/fairness/bert/response')
def Individual_UploadFile( 
                 text:Annotated[str, Form()], auth= Depends(auth)
                 ):
    try:
        bertService=BertService()
        log.info("before invoking service ")
        log.debug("request payload: " + str(text))
        log.debug(str(text))
        response = bertService.predict(text)
        log.debug(response)
        return response
    except FairnessException as cie:
        log.error(cie.__dict__)
        log.info("exit method")
        raise HTTPException(**cie.__dict__)

#Preprocessing mitigation
@Preprocessing_mitigation.post('/fairness/preprocessing/mitigation/get_data')
def get_preprocess_labels(MitigationType: Annotated[str, Form()],
            MitigationTechnique: Annotated[str, Form()],
            taskType: Annotated[str, Form()],
            fileId: Annotated[str, Form()], auth= Depends(auth)):
    payload = {"MitigationType": MitigationType,
            "MitigationTechnique": MitigationTechnique,
            "taskType": taskType,
            "fileId": fileId}
    try:
        log.info("before invoking create usecase service ")
        log.debug("request payload: " + str(payload))
        log.debug(str(payload))
        response = uiServicepreproc.upload_file_pretrainMitigation(payload)
        log.debug(response)
        return response
    except FairnessException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)



@Preprocessing_mitigation.post('/fairness/preprocessing/mitigation/BatchId')
def get_batchId(payload: BatchId, auth= Depends(auth)):
    log.info("Entered create usecase routing method")
    try:
        log.info("before invoking create usecase service ")
        log.debug("request payload: " + str(payload))
        response = uiServicepreproc.return_pretrainMitigation_protected_attrib(payload)
        log.info("after invoking create usecase service ")
        log.debug("response : " + str(response))
        log.info("exit create usecase routing method")
        return response
    except FairnessException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)

#Model Mitigation
@model_mitigation_router.post('/fairness/mitigation/model/UPFiles')
def mitigation_model_upload_files_Demo(
    trainingDatasetID: Annotated[str, Form()],
    modelID: Annotated[str, Form()], auth= Depends(auth)
): 
    payload={
        'trainingDatasetID': trainingDatasetID,
        'modelID': modelID
    }

    try:
        log.info('before invoking mitigation model upload files service')

        log.debug('request payload: '+str(payload))
        
        response=modelMitigate.mitigation_modelUpload_Fs(payload)

        log.debug('response: '+str(response))

        return response
    except FairnessException as cie:
        log.error(cie.__dict__)

        log.info('exit mitigation model upload files service')

        raise HTTPException(**cie.__dict__)
    

@model_mitigation_router.post('/fairness/mitigation/model/analyse')
def modelanalyze(payload: BatchId, auth= Depends(auth)):
    log.info("Entered create usecase routing method")
    try:
        log.info("before invoking create usecase service ")
        log.debug("request payload: " + str(payload))
        response = modelMitigate.mitigation_getmitigated_modelname_analyze(payload)
        log.info("after invoking create usecase service ")
        log.debug("response : " + str(response))
        log.info("exit create usecase routing method")
        return response
    except FairnessException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)

    


#analyse with group, individual metrics
@Analyze.post('/fairness/bias/Workbench/analyse/uploadfile')
def get_data(biasType: Annotated[str, Form()],
            methodType: Annotated[str, Form()],
            taskType: Annotated[str, Form()],
            fileId: Annotated[str, Form()], auth= Depends(auth)):
    
    payload = {"biasType": biasType,
            "methodType": methodType,
            "taskType": taskType,
            "fileId": fileId}
    try:
        log.info("before invoking create usecase service ")
        log.debug("request payload: " + str(payload))
        log.debug(str(payload))
        response = uiServicepreproc.analyse_UploadFile(payload)
        log.debug(response)
        return response
    except FairnessException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)

@Analyze.post('/fairness/bias/workbench/analyze/batchId/Attributes')
def analyze(payload: BatchId, auth= Depends(auth)):
    log.info("Entered create usecase routing method")
    try:
        log.info("before invoking create usecase service ")
        log.debug("request payload: " + str(payload))
        response = uiServicepreproc.return_protected_attrib_analyseDB(payload)
        log.info("after invoking create usecase service ")
        log.debug("response : " + str(response))
        log.info("exit create usecase routing method")
        return response
    except FairnessException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)


    





