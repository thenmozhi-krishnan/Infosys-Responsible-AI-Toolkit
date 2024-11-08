'''
MIT license https://opensource.org/licenses/MIT
Copyright 2024 Infosys Ltd
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''



import io
import os
import numpy as np
import pickle
from keras.models import load_model
import tempfile
import datetime,time
import pandas as pd
import json
import os
import zipfile
import shutil
import pdfkit
import requests

import matplotlib.pyplot as plt
import base64

from sklearn.metrics import accuracy_score

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PyPDF2 import PdfWriter, PdfReader

from app.config.urls import UrlLinks as UL
from sklearn.svm import SVC
from app.dao.Batch import Batch
from app.dao.ModelDb import Model
from app.dao.ModelAttributesDb import ModelAttributes
from app.dao.ModelAttributesValuesDb import ModelAttributesValues
from app.dao.DataDb import Data
from app.dao.DataAttributesDb import DataAttributes
from app.dao.DataAttributesValuesDb import DataAttributesValues
from app.dao.SaveFileDB import FileStoreDb

from app.dao.Security.SecReportDb import SecReport

from tensorflow.keras.preprocessing import image
from art.estimators.classification import SklearnClassifier
from art.attacks.poisoning.poisoning_attack_svm import PoisoningAttackSVM
from art.estimators.classification.scikitlearn import SklearnAPIClassifier
import concurrent.futures as con
from app.config.logger import CustomLogger

log =CustomLogger()


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

fetch_file = os.getenv("AZURE_GET_API")
upload_file = os.getenv("AZURE_UPLOAD_API")
dataset_container = os.getenv("DATA_CONTAINER_NAME")
model_container = os.getenv("MODEL_CONTAINER_NAME")
zip_container = os.getenv("ZIP_CONTAINER_NAME")
telemetry_flg =os.getenv("TELEMETRY_FLAG")

apiEndPoint ='/v1/security/model'
errorRequestMethod = 'GET'

class Utility:

    ArtAttackTypes = {
        'Evasion':[
                    "ProjectedGradientDescentTabular","ZerothOrderOptimization","QueryEfficient", 
                    "Deepfool", "Wasserstein", "Boundary", 'CarliniL2Method', 'Pixel', 
                    'UniversalPerturbation', 'FastGradientMethod', 'SpatialTransformation', 'Square', 
                    'ProjectGradientDescentImage', 'BasicIterativeMethod', 'SaliencyMapMethod',
                    'DecisionTree', 'IterativeFrameSaliency','SimBA', 'NewtonFool',
                    'ElasticNet','Poisoning','HopSkipJumpTabular',"HopSkipJumpImage", 
                    "QueryEfficientGradientAttackEndPoint", 'BoundaryAttackEndPoint',
                    'HopSkipJumpAttackEndPoint','VirtualAdversarialMethod', 'GeometricDecisionBasedAttack',
                    'Threshold'
                ],
        'Inference':[ 
                    'MembershipInferenceRule', 'AttributeInference',
                    'InferenceLabelOnlyGap', 'AttributeInferenceWhiteBoxDecisionTree',
                    'AttributeInferenceWhiteBoxLifestyleDecisionTree',
                    'LabelOnlyDecisionBoundary','MembershipInferenceBlackBox',
                    'LabelOnlyGapAttackEndPoint',
                    'MembershipInferenceBlackBoxRuleBasedAttackEndPoint',
                    'LabelOnlyDecisionBoundaryAttackEndPoint',
                    'MembershipInferenceBlackBoxAttackEndPoint'
                ]
    }

    def find_duplicates(x_train):

        """
        Returns an array of booleans that is true if that element was previously in the array
        :param x_train: training data
        :type x_train: `np.ndarray`
        :return: duplicates array
        :rtype: `np.ndarray`
        """
        
        try:
            dup = np.zeros(x_train.shape[0])
            for idx, x in enumerate(x_train):
                dup[idx] = np.isin(x_train[:idx], x).all(axis=1).any()
            return dup 
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "find_duplicates", e, apiEndPoint, errorRequestMethod)
 

    def get_adversarial_examples(x_train, y_train, attack_idx, x_val, y_val, kernel):
        try:
            # Create ART classifier for scikit-learn SVC
            art_classifier = SklearnClassifier(model=SVC(kernel=kernel), clip_values=(0, 10))
            art_classifier.fit(x_train, y_train)
            init_attack = np.copy(x_train[attack_idx])
            y_attack = np.array([1, 1]) - np.copy(y_train[attack_idx])
            attack = PoisoningAttackSVM(art_classifier, 0.0001, 1.0, x_train, y_train, x_val, y_val, max_iter=50)
            final_attack, _ = attack.poison(np.array([init_attack]), y=np.array([y_attack]))
            del y_attack,attack
            return final_attack, art_classifier 
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "get_adversarial_examples", e, apiEndPoint, errorRequestMethod)   


    def calc_precision_recall(predicted, actual, positive_value=1):
        try:
            score = 0  # both predicted and actual are positive
            num_positive_predicted = 0  # predicted positive
            num_positive_actual = 0  # actual positive
            for i in range(len(predicted)):
                if predicted[i] == positive_value:
                    num_positive_predicted += 1
                if actual[i] == positive_value:
                    num_positive_actual += 1
                if predicted[i] == actual[i]:
                    if predicted[i] == positive_value:
                        score += 1

            if num_positive_predicted == 0:
                precision = 1
            else:
                precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
            if num_positive_actual == 0:
                recall = 1
            else:
                recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

            return precision, recall
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "calc_precision_recall", e, apiEndPoint, errorRequestMethod)
    

    def readModelFile(payload):
        try:
            root_path = os.getcwd()
            root_path = Utility.getcurrentDirectory() + "/database"
            dirList = ["cacheMemory","data","model","payload","report"]
            for dir in dirList:
                dirs = root_path + "/" + dir
                if not os.path.exists(dirs):
                    os.mkdir(dirs)

            # Reading modelId from Batch Table
            batchList = Batch.findall({'BatchId':payload})[0]
            modelList = Model.findall({'ModelId':batchList['ModelId']})[0]
            modelName = modelList['ModelName']
            
            #  Reading ModelAttribute file content from MongoDB and stroing in Payload Folder.
            attributesData = {}
            attributeValues = ModelAttributesValues.findall({"ModelId":modelList['ModelId']})
            for value in attributeValues:
                attributes = ModelAttributes.findall({"ModelAttributeId":value.ModelAttributeId})[0]
                attributesData[attributes["ModelAttributeName"]] = value.ModelAttributeValues
            
            if attributesData['useModelApi'] == 'No':
                
                #  Reading model file content from MongoDB and stroing in Database Folder.
                if(os.getenv("DB_TYPE") == "mongo"):
                    modelDataId = modelList['ModelData']
                    modelFile = FileStoreDb.fs.get(modelDataId)
                    modelF = modelFile.read()
                    modelFileType = modelFile.filename.split('.')[-1]
                else:
                    modelFile =requests.get(url =fetch_file, params ={"container_name":model_container,"blob_name":modelList['ModelData']})
                    binary_data = modelFile.content
                    temp = io.BytesIO(binary_data)
                    modelF = temp.read()

                    content_disposition = modelFile.headers.get('content-disposition')
                    modelFileType = content_disposition.split(';')[1].split('=')[1].split('.')[-1]

                model_path = root_path + "/model"
                model_path = os.path.join(model_path,modelName+'.'+modelFileType)
                if os.path.exists(model_path):
                    os.remove(model_path)
                with open(model_path, 'wb') as f:
                    f.write(modelF)
                
                model_data:any
                if modelFileType == 'h5':
                    model_data = load_model(model_path) 
                else:
                    model_data = pickle.load(open(model_path, "rb"))         

                del modelF
                return model_data, model_path, modelName
            else:
                return modelName
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "readModelFile", e, apiEndPoint, errorRequestMethod)
    

    def readDataFile(payload):
        try:
            root_path = os.getcwd()
            root_path = Utility.getcurrentDirectory() + "/database"
            dirList = ["cacheMemory","data","model","payload","report"]
            for dir in dirList:
                dirs = root_path + "/" + dir
                if not os.path.exists(dirs):
                    os.mkdir(dirs)

            # Reading modelId from Batch Table
            batchList = Batch.findall({'BatchId':payload})[0]
            dataList = Data.findall({'DataId':batchList['DataId']})[0]
            modelList = Model.findall({'ModelId':batchList['ModelId']})[0]
            datasetName = modelList['ModelName'] 

            #  Reading Data file content from MongoDB and stroing in Database Folder.
            sampleDataId = dataList['SampleData']
            if(os.getenv("DB_TYPE") == "mongo"):
                dataFile = FileStoreDb.fs.get(sampleDataId)
                dataF = dataFile.read()
                dataFileType = dataFile.filename.split('.')[-1]
            else: 
                dataFile =requests.get(url =fetch_file, params ={"container_name":dataset_container,"blob_name":dataList['SampleData']})
                binary_data = dataFile.content
                temp = io.BytesIO(binary_data)
                dataF = temp.read()

                content_disposition = dataFile.headers.get('content-disposition')
                dataFileType = content_disposition.split(';')[1].split('=')[1].split('.')[-1]             
            
            data_path = root_path + "/data"
            data_path = os.path.join(data_path,datasetName + '.' + dataFileType)
            
            if os.path.exists(data_path):    
                os.remove(data_path)                                       
            with open(data_path, 'wb') as f:
                f.write(dataF)
            
            raw_data:any
            if dataFileType != 'csv':
                raw_data = image.load_img(data_path, target_size=(299, 299))
            else:
                raw_data = pd.read_csv(data_path)
            
            del dataF
            return raw_data, data_path
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "readDataFile", e, apiEndPoint, errorRequestMethod)
    

    def readPayloadFile(payload):
        try:
            root_path = os.getcwd()
            root_path = Utility.getcurrentDirectory() + "/database"
            dirList = ["cacheMemory","data","model","payload","report"]
            for dir in dirList:
                dirs = root_path + "/" + dir
                if not os.path.exists(dirs):
                    os.mkdir(dirs)
            
            # Reading modelId,dataId from Batch Table
            batchList = Batch.findall({'BatchId':payload})[0]
            modelList = Model.findall({'ModelId':batchList['ModelId']})[0]
            dataList = Data.findall({'DataId':batchList['DataId']})[0]
            payloadName = modelList['ModelName'] 
            
            #  Reading ModelAttribute content and stroing in attributesData.
            attributesData = {}
            attributeValues = ModelAttributesValues.findall({"ModelId":modelList['ModelId']})
            for value in attributeValues:
                attributes = ModelAttributes.findall({"ModelAttributeId":value.ModelAttributeId})[0]
                attributesData[attributes["ModelAttributeName"]] = value.ModelAttributeValues

            #  Reading DataAttribute content and stroing in attributesData1.
            attributesData1 = {}
            attributeValues = DataAttributesValues.findall({"DataId":dataList['DataId']})
            for value in attributeValues:
                attributes = DataAttributes.findall({"DataAttributeId":value.DataAttributeId})[0]
                attributesData1[attributes["DataAttributeName"]] = value.DataAttributeValues

            # combine attributesData1 and attributesData2 into one attributesData
            for k in attributesData1:
                if k not in attributesData:
                    attributesData[k] = attributesData1[k]

            modelEndPoint = modelList['ModelEndPoint'] 
            attributesData['modelEndPoint'] = modelEndPoint   
            payload_path = root_path + "/payload"
            payload_path = os.path.join(payload_path,payloadName+".txt") 
            if os.path.exists(payload_path):                          
                os.remove(payload_path)                                       
            with open(payload_path, 'w') as f:
                f.write(json.dumps(attributesData))
            return payload_path
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "readPayloadFile", e, apiEndPoint, errorRequestMethod)
    
    
    # def databaseWrite(payload):

    #     root_path = os.getcwd()
    #     root_path = Utility.getcurrentDirectory() + "/database"
    #     dirList = ["cacheMemory","data","model","payload","report"]
    #     for dir in dirList:
    #         dirs = root_path + "/" + dir
    #         if not os.path.exists(dirs):
    #             os.mkdir(dirs)

    #     modelList = Model.findall({'ModelName':payload})[0]

    #     #  Reading model file content from MongoDB and stroing in Database Folder.
    #     modelDataId = modelList['ModelDataID']
    #     modelFile = FileStoreDb.fs.get(modelDataId)
    #     modelF = modelFile.read()
    #     modelFileType = modelFile.filename.split('.')[-1]
    #     model_path = root_path + "/model"
    #     model_path = os.path.join(model_path,modelFile.filename)
    #     if os.path.exists(model_path):
    #         os.remove(model_path)
    #     with open(model_path, 'wb') as f:
    #         f.write(modelF)

    #     #  Reading Data file content from MongoDB and stroing in Database Folder.
    #     sampleDataId = modelList['SampleDataID']
    #     # dataFile = FileStoreDb.fs.find_one({'_id':sampleDataId})
    #     dataFile = FileStoreDb.fs.get(sampleDataId)
    #     dataF = dataFile.read()
    #     data_path = root_path + "/data"
    #     data_path = os.path.join(data_path,dataFile.filename)
    #     if os.path.exists(data_path):    
    #         os.remove(data_path)                                       
    #     with open(data_path, 'wb') as f:
    #         f.write(dataF)

    #     # Reading modelfile and datafile base on type of file
        
    #     del dataF,modelF
    #     return modelFileType, model_path, data_path
    
    
    def databaseDelete(payload):
        try:
            if os.path.isfile(payload):
                os.remove(payload)
            elif os.path.isdir(payload):    
                shutil.rmtree(payload)
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "databaseDelete", e, apiEndPoint, errorRequestMethod)
        
    
    def updateCurrentID():
        try:
            root_path = os.getcwd()
            directories = root_path.split(os.path.sep)
            testFlag = False
            for i in range(len(directories)-1, -1, -1):
                if directories[i] == 'test':
                    testFlag = True
                    break
            if testFlag == True:
                src_index = directories.index("test")
                new_path = os.path.sep.join(directories[:src_index])
            else:
                new_path = os.getcwd()
            path = new_path + '/app/config/urls.py'
    
            with open(path, 'r', encoding="utf8") as f:
                data = f.readlines()
    
            UL.Current_ID += 2  # Increment the Current_ID by 2
            Current_Report_ID = UL.Current_ID
            data[20] = f"    Current_ID = {Current_Report_ID}\n"
    
            # with open(path, 'w', encoding="utf8") as f1:
            #     f1.writelines(data)
    
            # Persist the updated UL.Current_ID value
            with open(path, 'r', encoding="utf8") as f2:
                updated_data = f2.read()
            with open(path, 'w', encoding="utf8") as f3:
                f3.write(updated_data.replace(f"    Current_ID = {Current_Report_ID - 2}", f"    Current_ID = {Current_Report_ID}"))
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "updateCurrentID", e, apiEndPoint, errorRequestMethod)    
    

    def updateReportsList(payload):
        try:
            # Taking latest report among reports
            latestDict = {}
            for d in payload['reportList']:
                name = d['ReportName'].split('.')[0]
                if name in payload['attackList']:
                    if d['ReportName'] == payload['modelName']+".zip":
                        continue 
                    else:
                        if name in latestDict:
                            if d['CreatedDateTime'] > latestDict[name]['CreatedDateTime']:
                                latestDict[name] = d
                        else:
                            latestDict[name] = d
            latestReportList = list(latestDict.values())
            
            # Sorting reports base on attack-type
            # reportList_dict = {item['ReportName']: item for item in reportList}
            # reportList = [reportList_dict[item] for item in payload['attackList'] if item in reportList_dict]
            # reportList = [next(item for item in latestReportList if item['ReportName'].split('.')[0] == attackName) for attackName in payload['attackList']]
            reportList = [report for attack in payload['attackList'] for report in latestReportList if attack == report['ReportName'].split('.')[0]]
            # reportList = []
            # for attackName in payload['attackList']:
            #     for report in latestReportList:
            #         if attackName == report['ReportName'].split('.')[0]:
            #             reportList.append(report)
            #             break
            #         else:
            #             continue

            return reportList
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "updateReportsList", e, apiEndPoint, errorRequestMethod)

    def sortReportsList(payload):

        # sort_reports = sorted(payload, key=
        #         lambda x:datetime.datetime.strptime(
        #             x['CreatedDateTime'].strftime("%Y-%m-%dT%H:%M:%S.%f"), "%Y-%m-%dT%H:%M:%S.%f"
        #         ), reverse=True)

        # OR
        try:
            sort_reports = sorted(payload, key=lambda x:x['CreatedDateTime'], reverse=True)

            return sort_reports
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "sortReportsList", e, apiEndPoint, errorRequestMethod)
    

    def dateTimeFormat(payload):
        try:
            # payload = "2023-12-27T18:22:10.015+00:00"
            # parsed_date = time.mktime(time.strptime(payload[:-6], "%Y-%m-%dT%H:%M:%S.%f"))
            # format_date = time.strftime("%d-%m-%Y %I:%M:%S %p", time.localtime(parsed_date))
            # print(format_date)

            # payload = datetime.datetime.now()
            format_date = payload.strftime("%d-%m-%Y %I:%M:%S %p")
            formatted_date = datetime.datetime.strptime(format_date, "%d-%m-%Y %I:%M:%S %p")

            # print('Parsed_Date', payload, type(payload))
            # print('Format_Date', format_date, type(format_date))
            # print('Formatted_Date', formatted_date, type(formatted_date))

            return format_date 
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "dateTimeFormat", e, apiEndPoint, errorRequestMethod)
    

    def combineList(payload):
        
        try:
            a = payload['attack_data'].tolist()
            b = payload['target_data'].tolist()
            c = [x + y for x,y in zip(a,b)]
            d = payload['prediction_data'].tolist()
            e = [x + [y] for x,y in zip(c,d)]
            f = []
            if payload['type'] == 'Evasion':
                for i in range(len(e)):
                    if e[i][-1] != e[i][-2]:
                        f.append([(i+2),e[i][-2],e[i][-1],'True'])
                        e[i].append('True')
                    else:
                        e[i].append('False')

                del a,b,c,d
                return e,f
            elif payload['type'] == 'Inference':
                for i in range(len(e)):
                    if e[i][-1] == e[i][-2]:
                        f.append([(i+2),e[i][-2],e[i][-1],'True'])
                        e[i].append('True')
                    else:
                        e[i].append('False')
                del a,b,c,d
                return e,f
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "combineList", e, apiEndPoint, errorRequestMethod)
    

    def checkList(payload):
        try:
            model = payload['model']
            x = payload['original_data']
            adversial_data = payload['adversial_data']
            prediction_list = []

            for i in range(x.shape[0]):
                l=[]
                l.append(x[i])
                l=np.array(l)
                pred=model.predict(l)
                label_benign = np.argmax(pred, axis=1)[0]

                m=[]
                m.append(adversial_data[i])
                m=np.array(m)
                pred=model.predict(m)
                label_adv = np.argmax(pred, axis=1)[0]

                if label_benign == label_adv:
                    continue
                else:
                    prediction_list.append([i, label_benign, label_adv, 'True'])

            del model,adversial_data
            return prediction_list
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "checkList", e, apiEndPoint, errorRequestMethod)
    

    def combineReportFile(payload):

        try:
            root_path = os.getcwd()
            root_path = Utility.getcurrentDirectory() + "/database"
            dirList = ["data","model","payload","report"]
            for dir in dirList:
                dirs = root_path + "/" + dir
                if not os.path.exists(dirs):
                    os.mkdir(dirs)
        
            reportList = SecReport.findall({"BatchId":payload['batchid']})
            reportList = Utility.updateReportsList({'reportList':reportList, 'modelName':payload['modelName'],'attackList':payload['attackList']})

            count = 0
            for i in range(len(reportList)):
                count = count + 1
                if(os.getenv("DB_TYPE") == "mongo"):
                    dataContent = FileStoreDb.findOne(reportList[i]["SecReportId"])
                    data = dataContent["data"]
                    attackname = dataContent['fileName'].split('.')[0]
                    fileName = dataContent["fileName"]
                else:
                    dataFile = requests.get(url = fetch_file, params ={"container_name":zip_container,"blob_name":reportList[i]["SecReportId"]})
                    binary_data = dataFile.content
                    temp = io.BytesIO(binary_data)
                    data = temp.read()
                    attackname = reportList[i]["SecReportId"].split('_')[0]
                    content_disposition = dataFile.headers.get('content-disposition')
                    fileName = content_disposition.split(';')[1].split('=')[1].split('.')[0]
                
                data_path = root_path + "/data"
                data_path = os.path.join(data_path,fileName) 
                if os.path.exists(data_path):                          
                    os.remove(data_path)                                       
                with open(data_path, 'wb') as f:
                    f.write(data)
                
                with zipfile.ZipFile(data_path, 'r') as zip_file:
                    for file_info in zip_file.infolist():
                        if file_info.filename.endswith('.html'):
                            with zip_file.open(file_info.filename) as data_file:
                                data = data_file.read().decode('utf-8')
                                # removing <style-tag> from .html file
                                start_style_tag = data.find('<style>')
                                end_style_tag = data.find('</style>') + len('</style>')
                                if start_style_tag != -1 and end_style_tag != -1:
                                    data_without_style = data[:start_style_tag] + data[end_style_tag:]
                                else:
                                    data_without_style = data
                            with open(os.path.join(payload['report_path'],f"report.html"), 'a', encoding='utf-8') as f:
                                # f.writelines(f"<h2 style='text-align:center; background-color:#8626C3; color:white'>{attackname}_Attack</h2>")
                                f.writelines(data_without_style)
                                # f.writelines('<br><br><br><br><br><br><br>')
                        elif file_info.filename.endswith('.csv'): 
                            with zip_file.open(file_info.filename, 'r') as csv_file:
                                with open(os.path.join(payload['report_path'],f"{attackname}.csv"), 'wb') as f:
                                    shutil.copyfileobj(csv_file, f)   
                        elif file_info.filename.endswith('.png'): 
                            # print(file_info.filename)
                            with zip_file.open(file_info.filename, 'r') as image_file:
                                # with open(os.path.join(payload['report_path'],f"{attackname}.png"), 'wb') as f:
                                with open(os.path.join(payload['report_path'],file_info.filename), 'wb') as f:
                                    shutil.copyfileobj(image_file, f)         
                os.remove(data_path)
            return count

        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "combineReportFile", e, apiEndPoint, errorRequestMethod)
            raise Exception


    def attackDesc(payload):
        try:
            desc = {

                "Poisoning":"""
                            Close implementation of poisoning attack on Support Vector Machines (SVM) by 
                            Biggio et al. 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1206.6389">Click here</a>
                            </p>
                        """,

                "MembershipInferenceRule":"""
                            Implementation of a simple, rule-based black-box membership inference attack. 
                            This implementation uses the simple rule: if the model's prediction for a 
                            sample is correct, then it is a member. Otherwise, it is not a member.
                        """,

                "MembershipInferenceBlackBox":"""
                            Implementation of a learned black-box membership inference attack. This 
                            implementation can use as input to the learning process probabilities/logits 
                            or losses, depending on the type of model and provided configuration.
                        """,

                "LabelOnlyDecisionBoundary":"""
                            Implementation of Label-Only Inference Attack based on Decision Boundary. You 
                            only need to call ONE of the calibrate methods, depending on which attack you 
                            want to launch.
                            <p class='content'>
                                Paper-1 link: <a href="https://arxiv.org/abs/2007.14321">Click here</a> (Choquette-Choo et al.)
                            </p>
                            <p class='content'>
                                Paper-2 link: <a href="https://arxiv.org/abs/2007.15528">Click here</a> (Li and Zhang)
                            </p>
                        """,

                "AttributeInferenceWhiteBoxLifestyleDecisionTree":"""
                            Implementation of Fredrikson et al. white box inference attack for decision 
                            trees. Assumes that the attacked feature is discrete or categorical, with 
                            limited number of possible values. For example: a boolean feature. 
                            <p class='content'>
                                Paper link: <a href="https://dl.acm.org/doi/10.1145/2810103.2813677">Click here</a>
                            </p>
                        """,

                "AttributeInferenceWhiteBoxDecisionTree":"""
                            A variation of the method proposed by of Fredrikson et al. in: 
                            <a href="https://dl.acm.org/doi/10.1145/2810103.2813677">Here</a>

                            Assumes the availability of the attacked model's predictions for the samples 
                            under attack, in addition to access to the model itself and the rest of the 
                            feature values. If this is not available, the true class label of the samples 
                            may be used as a proxy. Also assumes that the attacked feature is discrete or 
                            categorical, with limited number of possible values. For example: a boolean 
                            feature. 
                            <p class='content'>
                                Paper link: <a href="https://dl.acm.org/doi/10.1145/2810103.2813677">Click here</a>
                            </p>
                        """,

                "InferenceLabelOnlyGap":"""
                            Implementation of Label-Only Inference Attack based on Decision Boundary. You 
                            only need to call ONE of the calibrate methods, depending on which attack you 
                            want to launch.
                            <p class='content'>
                                Paper-1 link: <a href="https://arxiv.org/abs/2007.14321">Click here</a> (Choquette-Choo et al.)
                            </p>
                            <p class='content'>
                                Paper-2 link: <a href="https://arxiv.org/abs/2007.15528">Click here</a> (Li and Zhang)
                            </p>
                        """,

                "AttributeInference":"""
                            Implementation of a simple black-box attribute inference attack. The idea is 
                            to train a simple neural network to learn the attacked feature from the rest 
                            of the features and the model's predictions. Assumes the availability of the 
                            attacked model's predictions for the samples under attack, in addition to the 
                            rest of the feature values. If this is not available, the true class label of 
                            the samples may be used as a proxy.
                        """,
                
                "CarliniL2Method":"""
                            The CarliniL2Method attack is a type of adversarial attack designed to craft 
                            adversarial examples for machine learning models, particularly those used in
                            image classification. This attack was introduced by "Nicholas Carlini" and
                            "David Wagner" in their paper "Towards Evaluating the Robustness of Neural
                            Network" in 2017.

                            In this context of Adversarial Robustness Toolbox(ART), the CarliniL2Method 
                            is one of the attack methods provided for evaluating the robustness of machine
                            learning models. The goal of this attack is to find a small perturbation to
                            input data causes the model to misclassify the perturbation input while keeping
                            the perturbation within a specified L2 norm constraint.

                            The L_2 optimized attack of Carlini and Wagner (2016). This attack is among 
                            the most effective and should be used among the primary attacks to evaluate 
                            potential defences. A major difference wrt to the original implementation 
                            (<a href="https://github.com/carlini/nn_robust_attacks">Here</a>) is that we use 
                            line search in the optimization of the attack objective. 

                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1608.04644">Click here</a>
                            </p>
                        """,

                "Deepfool":"""
                            A Deepfool attack is a specified type of adversarial attack aimed at deep neural
                            networks. Deepfool is considered to be a white-box attack, meaning it assumes knowledge of
                            the target model's architecture and parameters. 

                            Implementation of the attack from Moosavi-Dezfooli et al. (2015). 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1511.04599">Click here</a>
                            </p>
                        """,

                "Boundary":"""
                            In the context of Adversarial Robustness Toolbox(ART), a boundary attack is a
                            type of attack strategy used to find adversarial examples. Adversarial examples
                            are inputs to a machine learning model that are intentionally designed to mis-
                            lead the model, causing it to make incorrect predictions.

                            The goal of a boundary attack is to find a data point close to the decision
                            boundary of the model, where a small perturbation can lead to a different class-
                            ification. This is achieved by iteratively moving the input towards the decision
                            boundary while keeping the perturbation small. The attack aims to find the minimal
                            perturbation needed to change the model's prediction.

                            Implementation of the boundary attack from Brendel et al. (2018). This is a 
                            powerful black-box attack that only requires final class prediction. 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1712.04248">Click here</a>
                            </p>
                        """,

                "UniversalPerturbation":"""
                            A Universal-Perturbation attack refers to the creation of a single perturbation
                            that, when added to different inputs, causes a model to misclassify them. This
                            perturbation is crafted to be universal in the sense that it can be applied to
                            various inputs, leading to consistent misclassifications.

                            Universal perturbation are partically concerning as they can generalize across
                            a wide range of input samples, making the model vulnerable to misclassifying
                            diverse data points. Adversarial robustness techniques, such as those provided
                            by ART, aim to defend against such attacks by improving the model's resilience
                            to adversarial perturbations.

                            Implementation of the attack from Moosavi-Dezfooli et al. (2016). Computes a 
                            fixed perturbation to be applied to all future inputs. To this end, it can use 
                            any adversarial attack method. 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1610.08401">Click here</a> 
                            </p>
                        """,

                "FastGradientMethod":"""
                            In the field of adversarial machine learning, gradient-based attacks are a class
                            of techniques that leverage the gradients of a neural network's loss function with
                            respect to its inputs. The Fast Gradient Method is a well-known example of a gradient
                            based attack. It involves perturbing input data by a small amount in the direction that
                            maximizes the loss, leading to misclassification by the model.

                            This attack was originally implemented by Goodfellow et al. (2015) with the 
                            infinity norm (and is known as the “Fast Gradient Sign Method”). This 
                            implementation extends the attack to other norms, and is therefore called the 
                            Fast Gradient Method. 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1412.6572">Click here</a>  
                            </p>
                        """,

                "SpatialTransformation":"""
                            Implementation of the spatial transformation attack using translation and 
                            rotation of inputs. The attack conducts black-box queries to the target model 
                            in a grid search over possible translations and rotations to find optimal attack 
                            parameters. 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1712.02779">Click here</a> 
                            </p>
                        """,

                "Pixel":"""
                            This attack was originally implemented by Vargas et al. (2019). It is 
                            generalisation of One Pixel Attack originally implemented by Su et al. (2019).
                            One Pixel Attack Paper link: <a href="https://arxiv.org/abs/1710.08864"></a> 
                            Pixel Attack 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1906.06026">Click here</a> 
                            </p>                        
                        """,

                "Wasserstein":"""
                            Implements Wasserstein Adversarial Examples via Projected Sinkhorn Iterations 
                            as evasion attack. 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1902.07906">Click here</a>
                            </p>
                        """,

                "Square":"""
                            This class implements the SquareAttack attack. 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1912.00049">Click here</a>
                            </p>
                        """,

                "ProjectGradientDescentImage":"""
                            The Projected Gradient Descent attack is an iterative method in which, after 
                            each iteration, the perturbation is projected on an lp-ball of specified radius 
                            (in addition to clipping the values of the adversarial sample so that it lies 
                            in the permitted data range). This is the attack proposed by Madry et al. for 
                            adversarial training. 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1706.06083">Click here</a>
                            </p>
                        """,

                "BasicIterativeMethod":"""
                            The Basic Iterative Method is the iterative version of FGM and FGSM. 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1607.02533">Click here</a>
                            </p>
                        """,

                "SaliencyMapMethod":"""
                            Implementation of the Jacobian-based Saliency Map Attack (Papernot et al. 2016).
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1511.07528">Click here</a>
                            </p>
                        """,

                "IterativeFrameSaliency":"""
                            Implementation of the attack framework proposed by Inkawhich et al. (2018). 
                            Prioritizes the frame of a sequential input to be adversarially perturbed based 
                            on the saliency score of each frame. 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1811.11875">Click here</a>
                            </p>
                        """,

                "SimBA":"""
                            This class implements the black-box attack SimBA. 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1905.07121">Click here</a>
                            </p>
                        """,

                "NewtonFool":"""
                            Implementation of the attack from Uyeong Jang et al. (2017). 
                            <p class='content'>
                                Paper link: <a href="http://doi.acm.org/10.1145/3134600.3134635">Click here</a>
                            </p>
                        """,

                "ElasticNet":"""
                            The elastic net attack of Pin-Yu Chen et al. (2018). 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1709.04114">Click here</a>
                            </p>
                        """,

                "QueryEfficient":"""
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1712.07113">Click here</a>
                            </p>
                        """,

                "ProjectedGradientDescentTabular":"""
                            The Projected Gradient Descent attack is an iterative method in which, after 
                            each iteration, the perturbation is projected on an lp-ball of specified radius 
                            (in addition to clipping the values of the adversarial sample so that it lies 
                            in the permitted data range). This is the attack proposed by Madry et al. for 
                            adversarial training. 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1706.06083">Click here</a>
                            </p>
                        """,

                "DecisionTree":"""
                            Close implementation of Papernot's attack on decision trees following Algorithm 
                            2 and communication with the authors. 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1605.07277">Click here</a>
                            </p>
                        """,

                "HopSkipJumpTabular":"""
                            Implementation of the HopSkipJump attack from Jianbo et al. (2019). This is a 
                            powerful black-box attack that only requires final class prediction, and is an 
                            advanced version of the boundary attack. 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1904.02144">Click here</a>
                            </p>
                        """,

                "ZerothOrderOptimization":"""
                            The black-box zeroth-order optimization attack from Pin-Yu Chen et al. (2018). 
                            This attack is a variant of the C&W attack which uses ADAM coordinate descent 
                            to perform numerical estimation of gradients. 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1708.03999">Click here</a>
                            </p>
                        """,

                "HopSkipJumpImage":"""
                            Implementation of the HopSkipJump attack from Jianbo et al. (2019). This is a 
                            powerful black-box attack that only requires final class prediction, and is an 
                            advanced version of the boundary attack. 
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1904.02144">Click here</a>
                            </p>
                        """,
                "QueryEfficientGradientAttackEndPoint":"""
                            <p class='content'>
                                Paper link: <a href="https://arxiv.org/abs/1712.07113">Click here</a>
                            </p>
                        """,
                "BoundaryAttackEndPoint":"""
                            In the context of Adversarial Robustness Toolbox(ART), a boundary attack is a
                            type of attack strategy used to find adversarial examples. Adversarial examples
                            are inputs to a machine learning model that are intentionally designed to mis-
                            lead the model, causing it to make incorrect predictions.

                            The goal of a boundary attack is to find a data point close to the decision
                            boundary of the model, where a small perturbation can lead to a different class-
                            ification. This is achieved by iteratively moving the input towards the decision
                            boundary while keeping the perturbation small. The attack aims to find the minimal
                            perturbation needed to change the model's prediction.

                        Implementation of the boundary attack from Brendel et al. (2018). This is a 
                        powerful black-box attack that only requires final class prediction. 
                        <p class='content'>
                            Paper link: <a href="https://arxiv.org/abs/1712.04248">Click here</a>
                        </p>
                    """,
            "HopSkipJumpAttackEndPoint":"""
                        Implementation of the HopSkipJump attack from Jianbo et al. (2019). This is a 
                        powerful black-box attack that only requires final class prediction, and is an 
                        advanced version of the boundary attack. 
                        <p class='content'>
                            Paper link: <a href="https://arxiv.org/abs/1904.02144">Click here</a>
                        </p>
                    """,
            "LabelOnlyGapAttackEndPoint":"""
                        Implementation of Label-Only Inference Attack based on Decision Boundary. You 
                        only need to call ONE of the calibrate methods, depending on which attack you 
                        want to launch.
                        <p class='content'>
                            Paper-1 link: <a href="https://arxiv.org/abs/2007.14321">Click here</a> (Choquette-Choo et al.)
                        </p>
                        <p class='content'>
                            Paper-2 link: <a href="https://arxiv.org/abs/2007.15528">Click here</a> (Li and Zhang)
                        </p>
                    """,
            "MembershipInferenceBlackBoxRuleBasedAttackEndPoint":"""
                        Implementation of a simple, rule-based black-box membership inference attack. 
                        This implementation uses the simple rule: if the model's prediction for a 
                        sample is correct, then it is a member. Otherwise, it is not a member.
                    """,
            "LabelOnlyDecisionBoundaryAttackEndPoint":"""
                        Implementation of Label-Only Inference Attack based on Decision Boundary. You 
                        only need to call ONE of the calibrate methods, depending on which attack you 
                        want to launch.
                        <p class='content'>
                            Paper-1 link: <a href="https://arxiv.org/abs/2007.14321">Click here</a> (Choquette-Choo et al.)
                        </p>
                        <p class='content'>
                            Paper-2 link: <a href="https://arxiv.org/abs/2007.15528">Click here</a> (Li and Zhang)
                        </p>
                    """,
            "MembershipInferenceBlackBoxAttackEndPoint":"""
                        Implementation of a learned black-box membership inference attack. This 
                        implementation can use as input to the learning process probabilities/logits 
                        or losses, depending on the type of model and provided configuration.
                    """,
            "VirtualAdversarialMethod":"""
                        Implementation of a VirtualAdversarialMethod evasion attack. VirtualAdversarialMethod
                        resembles adversarial training, but distinguishes itself in that it determines the 
                        adversarial direction from the model distribution alone without using the label 
                        information, making it applicable to semi-supervised learning.
                        <p class='content'>
                            Paper link: <a href="https://arxiv.org/abs/1507.00677">Click here</a>
                        </p>
                    """,
            "GeometricDecisionBasedAttack":"""
                        Implementation of a Geometric Decision-Based evasion attack. It is a black-box attack
                        that uses the decision boundary of the model to craft adversarial examples. It is also
                        a query-optimised attack, meaning it uses the minimum number of queries to the model to
                        generate perturbations.
                        <p class='content'>
                            Paper link: <a href="https://arxiv.org/abs/2003.06468">Click here</a>
                        </p>                    
            """,
            "Threshold":"""
                        Implementation of a Threshold evasion attack. Threshold
                        validate the dual quality assessment on state-of-the-art neural networks (WideResNet,
                        ResNet, AllConv, DenseNet, NIN, LeNet and CapsNet) as well as adversarial defences for 
                        image classification problem.
                        <p class='content'>
                            Paper link: <a href="https://arxiv.org/abs/1906.06026">Click here</a>
                        </p>
                    """
        }

            if payload in desc:
                return desc[payload]
            
            else:
                return ''
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "attackDesc", e, apiEndPoint, errorRequestMethod)

    
    def htmlCssContent1():
        try:
            html_data = """
                <style>
                    h2{
                        text-align: center;
                        background-color: #8626C3;
                        color: white;
                        font-size: 35px;
                    }
                    h3{
                        color: darkorchid;
                    }


                    table{
                        border-collapse: collapse;
                        width: 100%;
                    }
                    table,th,td{
                        border: 2px solid #000000;
                    }
                    th{
                        background-color: #f2f2f2;
                    }


                    .header{
                        color: darkorchid;
                        font-size: 25px;
                        margin: inherit;
                        font-style: normal;
                        margin-top: 25px;
                    }


                    .note{
                        background-color: #f9f9f9;
                        padding: 10px;
                        border-left: 5px solid #ccc;
                        margin-bottom: 20px;
                        line-height: 1.6;
                    }
                    .content{
                        font-weight: normal;
                        color: #8626C3;
                        font-size: 20px; 
                        font-family: sans-serif;
                    }


                    .data-container{
                        width: 80%;
                        margin: 30px;
                    }
                    .data-label, .data-value{
                        display: inline-block;
                        width: 48%;
                        text-align: left;
                    }
                    .data-label{
                        margin-right: 15px;
                    }


                    .container{
                        padding: 20px;
                        margin-top: -20px;
                        font-size: 20px;
                    }
                    .face-card{
                        display: inline-block;
                        border: 1px solid #ccc;
                        padding: 20px;
                        width: calc(33.33% -20px);
                        text-align: center;
                        box-shadow: 0 8px 12px rgb(5,30,217,0.94);
                        margin-top: 10px;
                        margin-left: 45px;
                        margin-right: 5px;
                        width: 20%;
                        background-color: #e4eae5;
                    }
                    .success{
                        color: green;
                    }
                    .fail{
                        color: red;
                    }
                    .pending{
                        color: orange;
                    }

                    
                    #datetime{
                        text-align: right;
                        margin: 15px;
                        font-size: 20px;
                        color: #8626C3;
                    }

                    
                    .container {
                        padding: 20px;
                        margin-top: 15px;
                        font-size: 20px;
                    }
                    table {
                        border-collapse: collapse;
                        width: 100%;
                    }
                    th,
                    td {
                        text-align: left;
                        padding: 8px;
                    }
                    th {
                        background-color: #f2f2f2;
                    }
                    .pass-icon {
                        color: green;
                    }
                    .fail-icon {
                        color: red;
                    }
                </style>
            """

            return html_data
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "htmlCssContent1", e, apiEndPoint, errorRequestMethod)
    

    def htmlContent1(payload):
        try:
            if 'modelName' in payload:
                
                html_content = f"""
                        <body>

                            <div id='datetime'><strong>{Utility.dateTimeFormat(payload['reportTime'])}</strong></div>

                            <h2>REPORT</h2>

                            <h3 class="header"><strong>OBJECTIVE</strong></h3>
                            <div class='note'>
                                <p class="content">The report explores various models and their vulnerability to adversarial attacks. 
                                It delves into the methodologies of applying adversarial attacks on these models and provides 
                                insightful results regarding the effectiveness of such attacks. The findings shed light on the 
                                models' robustness or susceptibility, contributing valuable insights to the broader understanding 
                                of security in machine learning applications.</p>
                            </div>

                            <h3 class="header"><strong>MODEL INFORMATION</strong></h3>
                            <div class="data-container">
                                <div class="data-label content"><strong>Model Name:</strong></div>
                                <div class="data-value content">{payload['modelName']}</div>
                            </div>
                            <div class="data-container">
                                <div class="data-label content"><strong>Target DataType:</strong></div>
                                <div class="data-value content">{payload['model_metaData']['targetDataType']}</div>
                            </div>
                            <div class="data-container">
                                <div class="data-label content"><strong>Model Api:</strong></div>
                                <div class="data-value content">{payload['model_metaData']['useModelApi']}</div>
                            </div>
                            <div class="data-container">
                                <div class="data-label content"><strong>Model End Point:</strong></div>
                                <div class="data-value content">{payload['model_metaData']['modelEndPoint']}</div>
                            </div>
                            <div class="data-container">
                                <div class="data-label content"><strong>Target Output Classes:</strong></div>
                                <div class="data-value content">{payload['model_metaData']['groundTruthClassNames']}</div>
                            </div>
                            <div class="data-container">
                                <div class="data-label content"><strong>Target Classifier:</strong></div>
                                <div class="data-value content">{payload['model_metaData']['targetClassifier']}</div>
                            </div>
                            <div class="data-container">
                                <div class="data-label content"><strong>Target ColumnNames:</strong></div>
                                <div class="data-value content">{payload['model_metaData']['groundTruthClassLabel']}</div>
                            </div>
                            <hr>

                            <h3 class="header"><strong>ATTACK SUMMARY</strong></h3>
                            <div class="container">
                                <table id="attack-table">
                                    <thead>
                                        <tr>
                                            <th>Attack Type</th>
                                            <th>Attack Name</th>
                                            <th>Selected Attack</th>
                                            <th>Attack Accuracy</th>
                                            <th>Detection Accuracy<br>(XGB Model)</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {payload['rows']}
                                    </tbody>
                                </table>
                            </div>

                        </body>
                    """  
                
                if 'graph' in payload:
                    
                    html_content = html_content + f"<body><img src='data:image/png;base64,{payload['graph']}' alt='Attack Graph'></body>"
                    return html_content   
                else:
                    return html_content
            else:
                html_content = f"""
                            <h3 class="header"><strong>DESCRIPTION</strong></h3>
                            <div class="container">
                                <div class='note'>
                                    <p class='content'>{Utility.attackDesc(payload)}</p>
                                </div>
                            </div>
                        """

                return html_content
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "htmlContent1", e, apiEndPoint, errorRequestMethod)
    
    def htmlCssContent():
        try:
            html_data = """
                <style>
                    .heading-color {
                        color: rgb(28, 160, 242);
                    }

                    .heading-font {
                        font-family: math;
                    }

                    .text-color {
                        color: darkgray;
                    }

                    .heading-margin {
                        padding: 10px;
                        margin: 0px;
                    }

                    .report-container {
                        display: block;
                        font-family: Arial, sans-serif;
                    }

                    .datetime-container {
                        position: fixed;
                        top: 10px;
                        right: 10px;
                        font-size: 14px;
                        padding: 5px 10px;
                    }

                    .report-header {
                        display: block;
                        overflow: hidden;
                        padding: 10px;
                    }

                    .report-header h1 {
                        font-weight: bold;
                        position: relative;
                        font-size: 17px;
                    }

                    .report-header h1::after {
                        content: "";
                        position: absolute;
                        left: 0;
                        bottom: 0;
                        width: 7%;
                        height: 3px;
                        background-color: rgb(28, 160, 242);
                    }

                    .report-body {
                        display: block;
                    }

                    .report-section {
                        display: inline-block;
                        width: 49%;
                        box-sizing: border-box;
                        padding: 10px;
                        vertical-align: top;
                    }

                    .report-section-model {
                        display: inline-block;
                        width: 48%;
                        box-sizing: border-box;
                        margin-top: 0px;
                        vertical-align: top;
                    }

                    .remove-margin {
                        margin: 0px;
                    }

                    .bold-line {
                        border-top: 2px solid rgb(234, 228, 228);
                        font-weight: bold;
                        margin-top: 20px;
                        padding-top: 10px;
                    }

                    .attack-summary {
                        margin-top: 20px;
                        border-collapse: collapse;
                        width: max-content;
                        display: block;
                        overflow-x: auto;
                        white-space: nowrap;
                        margin-bottom: 30px;
                    }

                    .attack-summary th,
                    .attack-summary td {
                        padding: 8px;
                        text-align: left;
                        border-bottom: 1px solid #ddd;
                        min-width: 80px; 
                    }

                    .attack-summary th {
                        /* background-color: #f2f2f2; */
                    }

                    .attack-summary td {
                        color: rgb(124 116 116);
                    }

                    .attack-summary td:nth-child(3),
                    .attack-summary td:nth-child(4),
                    .attack-summary td:nth-child(5) {
                        /* text-align: center; */
                    }

                    .selected-attack {
                        font-weight: bold;
                        color: rgb(100, 232, 100);
                    }

                    .attack-accuracy,
                    .detection-accuracy {
                        display: inline-flex;
                        flex-direction: column;
                        align-items: center;
                        width: 100px;
                        margin-top: 5px;
                    }

                    .attack-accuracy-value,
                    .detection-accuracy-value {
                        font-size: 12px;
                        margin-bottom: 2px;
                    }

                    .attack-accuracy-bar,
                    .detection-accuracy-bar {
                        width: 100%;
                        height: 5px;
                        background-color: #e0e0e0;
                        position: relative;
                    }

                    .attack-accuracy-bar-fill,
                    .detection-accuracy-bar-fill {
                        position: absolute;
                        left: 0;
                        top: 0;
                        height: 100%;
                        background-color: rgb(127, 187, 224);
                    }

                    .attack-header {
                        display: block;
                        overflow: hidden;
                        padding: 10px;
                    }

                    .attack-header h2 {
                        font-weight: bold;
                        position: relative;
                    }

                    .attack-header h2::after {
                        content: "";
                        position: absolute;
                        left: 0;
                        bottom: 0;
                        width: 5%;
                        height: 3px;
                        background-color: rgb(102, 175, 220);
                    }

                    .attack-data-table {
                        margin-top: 20px;
                        border-collapse: collapse;
                        width: max-content;
                        display: block;
                        overflow-x: auto;
                        white-space: nowrap;
                    }

                    .attack-data-table th,
                    .attack-data-table td {
                        padding: 5px;
                        /* text-align: left; */
                        border-bottom: 1px solid #ddd;
                        min-width: 130px;
                    }

                    .attack-data-table th {
                        /* background-color: #f2f2f2; */
                    }

                    .attack-data-table td {
                        color: rgb(124 116 116);
                        /* font-size: 12px; */
                    }

                    .attack-data-table td:nth-child(3),
                    .attack-data-table td:nth-child(4),
                    .attack-data-table td:nth-child(5) {
                        /* text-align: center; */
                    }

                    .attack-data-table-column {
                        /* display: inline-flex;
                        flex-direction: column;
                        align-items: center;
                        width: 100px; */
                        margin-top: 5px;
                    }

                    .attack-data-table-column-value {
                        font-size: 12px;
                        text-align: center;
                        /* margin-bottom: 2px; */
                    }

                    .attack-data {
                        margin-top: 20px;
                        border-collapse: collapse;
                        width: max-content;
                        display: block;
                        overflow-x: auto;
                        white-space: nowrap;
                    }

                    .attack-data th,
                    .attack-data td {
                        padding: 5px;
                        /* text-align: left; */
                        border-bottom: 1px solid #ddd;
                        min-width: 180px; 
                    }

                    .attack-data th {
                        /* background-color: #f2f2f2; */
                    }

                    .attack-data td {
                        color: rgb(124 116 116);
                        text-align: center;
                        /* font-size: 12px; */
                    }

                    .attack-data td:nth-child(3),
                    .attack-data td:nth-child(4),
                    .attack-data td:nth-child(5) {
                        /* text-align: center; */
                    }

                    .graph-container {
                        width: 100%; 
                        height: 560px; 
                    }

                    .graph-container-attack {
                        width: 100%;
                        height: 220px;
                    }

                    .graph-image {
                        max-width: 100%; 
                        max-height: 100%; 
                    }
                </style>
            """

            return html_data
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "htmlCssContent", e, apiEndPoint, errorRequestMethod)
    
    def htmlContent(payload):
        try:
            if 'modelName' in payload:

                html_data = f"""
                    <body>
                        <div class="report-container">
                            <div class="datetime-container">
                                <span id="datetime">
                                    <p class="text-color">{Utility.dateTimeFormat(payload['reportTime'])}</p>
                                </span>
                            </div>
                            <div class="report-header">
                                <h1 style="font-family: system-ui; color: #706a6a;">MODEL ROBUSTNESS ASSESSMENT REPORT</h1>
                            </div>
                            <div>
                                <h2 class="heading-color heading-font heading-margin">OBJECTIVE</h2>
                            </div>
                            <div class="report-body">
                                <div class="report-section">
                                    <p class="text-color">The report explores various models and their vulnerability to adversarial attacks.
                                        It delves into the methodologies of applying adversarial attacks on these models and provides insightful results
                                        regarding the effectiveness of such attacks. The findings shed light on the models' robustness or
                                        susceptibility, contributing valuable insights to the broader understanding of security in machine
                                        learning applications.</p>
                                </div>
                                <div class="report-section">
                                    <div>
                                        <h3 class="heading-color">MODEL INFORMATION</h3>
                                    </div>
                                    <div class="report-body">
                                        <div class="report-section-model">
                                            <p class="remove-margin">Model Name:</p>
                                            <p class="remove-margin">Target DataType:</p>
                                            <p class="remove-margin">Model Api:</p>
                                            <p class="remove-margin">Model End Point:</p>
                                            <p class="remove-margin">Target Output Classes:</p>
                                            <p class="remove-margin">Target Classifier:</p>
                                            <p class="remove-margin">Target ColumnNames:</p>
                                        </div>
                                        <div class="report-section-model">
                                            <p class="remove-margin text-color">{payload['modelName']}</p>
                                            <p class="remove-margin text-color">{payload['model_metaData']['targetDataType']}</p>
                                            <p class="remove-margin text-color">{payload['model_metaData']['useModelApi']}</p>
                                            <p class="remove-margin text-color">{payload['model_metaData']['modelEndPoint']}</p>
                                            <p class="remove-margin text-color">{payload['model_metaData']['groundTruthClassNames']}</p>
                                            <p class="remove-margin text-color">{payload['model_metaData']['targetClassifier']}</p>
                                            <p class="remove-margin text-color">{payload['model_metaData']['groundTruthClassLabel']}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="bold-line"></div>

                            <div class="attack-summary">
                                <h3 class="heading-color heading-margin">ATTACK SUMMARY</h3>
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Attack Type</th>
                                            <th>Attack Name</th>
                                            <th>Selected Attack</th>
                                            <th>Attack Accuracy</th>
                                            <th>Detection Accuracy (XGB Model)</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {payload['rows']}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </body>
                """

                if 'graph' in payload:
                    
                    html_data = html_data + f"<body><div class='graph-container'><img src='data:image/png;base64,{payload['graph']}' alt='Attack Graph' class='graph-image'></div></body>"
                    return html_data   
                else:
                    return html_data
            else:
                return ''
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "htmlContent", e, apiEndPoint, errorRequestMethod)

       
    def htmlCssContentReport():
        try:
            html_data = """
                <style>
                    .heading-color {
                        color: rgb(28, 160, 242);
                    }

                    .heading-font {
                        font-family: math;
                    }

                    .text-color {
                        color: darkgray;
                    }

                    .heading-margin {
                        padding: 10px;
                        margin: 0px;
                    }

                    .report-container {
                        display: block;
                        font-family: Arial, sans-serif;
                    }

                    .report-body {
                        display: block;
                    }

                    .report-section {
                        display: inline-block;
                        width: 49%;
                        box-sizing: border-box;
                        padding: 10px;
                        vertical-align: top;
                    }

                    .remove-margin {
                        margin: 0px;
                    }

                    .attack-header {
                        display: block;
                        overflow: hidden;
                        padding: 10px;
                    }

                    .attack-header h2 {
                        font-weight: bold;
                        position: relative;
                    }

                    .attack-header h2::after {
                        content: "";
                        position: absolute;
                        left: 0;
                        bottom: 0;
                        width: 5%;
                        height: 3px;
                        background-color: rgb(102, 175, 220);
                    }

                    .attack-data-table {
                        margin-top: 20px;
                        border-collapse: collapse;
                        width: max-content;
                        display: block;
                        overflow-x: auto;
                        white-space: nowrap;
                    }

                    .attack-data-table th,
                    .attack-data-table td {
                        padding: 5px;
                        /* text-align: left; */
                        border-bottom: 1px solid #ddd;
                        min-width: 130px;
                    }

                    .attack-data-table th {
                        /* background-color: #f2f2f2; */
                    }

                    .attack-data-table td {
                        color: rgb(124 116 116);
                        /* font-size: 12px; */
                    }

                    .attack-data-table td:nth-child(3),
                    .attack-data-table td:nth-child(4),
                    .attack-data-table td:nth-child(5) {
                        /* text-align: center; */
                    }

                    .attack-data-table-column {
                        /* display: inline-flex;
                        flex-direction: column;
                        align-items: center;
                        width: 100px; */
                        margin-top: 5px;
                    }

                    .attack-data-table-column-value {
                        font-size: 12px;
                        text-align: center;
                        /* margin-bottom: 2px; */
                    }

                    .attack-data {
                        margin-top: 20px;
                        border-collapse: collapse;
                        width: max-content;
                        display: block;
                        overflow-x: auto;
                        white-space: nowrap;
                    }

                    .attack-data th,
                    .attack-data td {
                        padding: 5px;
                        /* text-align: left; */
                        border-bottom: 1px solid #ddd;
                        min-width: 180px; 
                    }

                    .attack-data th {
                        /* background-color: #f2f2f2; */
                    }

                    .attack-data td {
                        color: rgb(124 116 116);
                        text-align: center;
                        /* font-size: 12px; */
                    }

                    .attack-data td:nth-child(3),
                    .attack-data td:nth-child(4),
                    .attack-data td:nth-child(5) {
                        /* text-align: center; */
                    }

                    .graph-container {
                        width: 100%; 
                        height: 560px; 
                    }

                    .graph-container-attack {
                        width: 100%;
                        height: 220px;
                    }

                    .graph-image {
                        max-width: 100%; 
                        max-height: 100%; 
                    }
                </style>
            """

            return html_data
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "htmlCssContentReport", e, apiEndPoint, errorRequestMethod)
    
    def htmlContentReport(payload):
        try:
            html_data = f"""
                <body>
                    <div class="report-container">

                        <div class="attack-header">
                            <h2 class="heading-color heading-font">{payload['attackName']}_Attack</h2>
                        </div>

                        <div class="report-body">
                            <div class="report-section">
                                <h3 class="heading-color">Description</h3>
                                <p class="text-color">{Utility.attackDesc(payload['attackName'])}</p>
                            </div>
                            <div class="report-section">
                                <h3 class="heading-color">Visualization for Prediction Data</h3>
                                {payload['graph_html']}
                            </div>
                        </div>
                    </div>
                    <div class="attack-data-table">
                        <h3 class="heading-color heading-margin">Attack Status</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>Attack ID</th>
                                    <th>Model Name</th>
                                    <th>Attack Name</th>
                                    <th>Status</th>
                                    <th>Mean Difference</th>
                                </tr>
                            </thead>
                            <tbody>
                                {payload['attack_status_row']}
                            </tbody>
                        </table>
                    </div>
                    <div class="attack-data">
                        <h3 class="heading-color heading-margin">Adversial input used for attack and Adversial output generated</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>Sample_index</th>
                                    <th>Actual_labels </th>
                                    <th>Final_labels</th>
                                    <th>Success</th>
                                </tr>
                            </thead>
                            <tbody>
                                {payload['attack_ipop_row']}
                            </tbody>
                        </table>
                    </div>
                </body>
            """
            return html_data
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "htmlContentReport", e, apiEndPoint, errorRequestMethod)
    
    
    def htmlToPdfWithWatermark(payload):
        try:
            # convert html file into pdf file
            html_path = os.path.join(payload['folder_path'],"report.html")
            pdf_path = os.path.join(payload['folder_path'],"report.pdf")
            option = {
                'page-size':'A4',
                'orientation':'Portrait',
                # 'margin-top':'0.75in',
                # 'margin-right':'0.75in',
                # 'margin-bottom':'0.75in',
                # 'margin-left':'0.75in',
                'encoding':'UTF-8',
                'no-outline':None,
                'footer-center':'Page [page] of [toPage]',
            }
            pdfkit.from_file(html_path, output_path=pdf_path, options=option)

            # create watermark.pdf file
            watermark_path = os.path.join(payload['folder_path'], 'watermark.pdf')
            # txt = 'Infosys'
            txt = ''
            c = canvas.Canvas(watermark_path, pagesize=letter)
            c.setFont('Helvetica', 50)
            c.setFillColorRGB(0.53,0.15,0.76)
            c.setFillAlpha(0.13)
            c.rotate(45)
            c.drawString(400, 8, txt)
            c.save()

            # adding watermark in each page of report.pdf file
            combine_pdf = os.path.join(payload['folder_path'], 'report.pdf')
            modify_pdf = os.path.join(payload['folder_path'], 'report.pdf')

            with open(combine_pdf, 'rb') as pdf_file, open(watermark_path, 'rb') as watermark_file:
                pdf_reader = PdfReader(pdf_file)
                watermark_reader = PdfReader(watermark_file)
                watermark_page = watermark_reader.pages[0]
                pdf_writer = PdfWriter()

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page.merge_page(watermark_page)
                    pdf_writer.add_page(page)

                with open(modify_pdf, 'wb') as out_file:
                    pdf_writer.write(out_file)
            os.remove(watermark_path)

            return 
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "htmlToPdfWithWatermark", e, apiEndPoint, errorRequestMethod)
    

    # def checkAttackListStatus(payload):

    #     statusList = []

    #     for filename in os.listdir(payload['folder_path']):
    #         if filename.endswith('.csv'):
    #             csv_path = os.path.join(payload['folder_path'], filename)
    #             df = pd.read_csv(csv_path)
    #             col = df[df[df.columns[-1]] == True][df.columns[-1]].value_counts().tolist()
    #             if len(col) > 0:
    #                 statusList.append({filename.split('.')[0]:'Fail'})
    #             else:
    #                 statusList.append({filename.split('.')[0]:'Pass'})
    #         else:
    #             continue

    #     return statusList
    

    # def generateDefenceAccuracy(payload):

    #     try:
    #         payload_folder_path = Utility.getcurrentDirectory() + "/database/payload"
    #         payload_path = os.path.join(payload_folder_path,payload["modelName"] + ".txt")
    #         with open(f'{payload_path}') as f:
    #             data = f.read()
    #         data = json.loads(data)
    #         Output_column = data["groundTruthClassLabel"]

    #         #reading original dataset
    #         data_path = Utility.getcurrentDirectory() + f'/database/data/{payload["modelName"]}.csv'
    #         df1 = pd.read_csv(data_path, delimiter=',')
    #         df1.drop(Output_column,axis=1,inplace=True)
    #         df1.insert(df1.shape[1],"Attack",[0 for i in range(df1.shape[0])])

    #         #reading attack dataset
    #         df2 = pd.read_csv(payload['csv_path'])
    #         df2 = df2[df2[df2.columns[-1]] == True]
    #         df2.drop(Output_column,axis=1,inplace=True)
    #         df2.drop(columns=df2.columns[-2:], axis=1, inplace=True)
    #         df2.insert(df2.shape[1],"Attack",[1 for i in range(df2.shape[0])])
    
    #         #creating defensemodel dataset
    #         fields=[]
    #         for col in df1.columns:
    #             fields.append(col)
    #         df1=np.array(df1)
    #         df2=np.array(df2)

    #         #creating defense_model path
    #         # root_path = Utility.getcurrentDirectory() + "/database"
    #         # temp_path = root_path + "/cacheMemory"
    #         temp_path = Utility.getcurrentDirectory() + "/database/cacheMemory"
    #         temp_path = os.path.join(temp_path,f'{payload["modelName"]}defenseModel.csv')

    #         if os.path.exists(temp_path):
    #             os.remove(temp_path)
    #         with open(temp_path,"w",newline="") as f:
    #             write = csv.writer(f)
    #             write.writerow(fields)
    #             write.writerows(df1)
    #             write.writerows(df2)
        
    #         # creating defense Model
    #         df=pd.read_csv(temp_path)
    #         X=df.loc[:,df.columns!="Attack"]
    #         Y=df["Attack"]
    #         X_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1)
    #         model=XGBClassifier()
    #         model.fit(X_train,y_train)
    #         accuracy=model.score(x_test, y_test)

    #         del model,X_train,x_test,y_train,y_test,df1,df2
    #         Utility.databaseDelete(temp_path)

    #         return accuracy
    #     except Exception as exc:
    #         print(exc)


    def generateDefenceAccuracy1(payload):

        try:
            payload_folder_path = Utility.getcurrentDirectory() + "/database/payload"
            payload_path = os.path.join(payload_folder_path,payload["modelName"] + ".txt")
            with open(f'{payload_path}') as f:
                data = f.read()
            data = json.loads(data)
            Output_column = data["groundTruthClassLabel"]

            for filename in os.listdir(payload['folder_path']):
                if filename.endswith('.pkl'):
                    # load model file
                    model_file = os.path.join(payload['folder_path'], filename)
                    with open(model_file, 'rb') as file:
                        load_model = pickle.load(file)
                    # reading attack dataset 
                    df = pd.read_csv(payload['csv_path'])
                    df = df[df[df.columns[-1]] == True]
                    df.drop(Output_column,axis=1,inplace=True)
                    df.drop(columns=df.columns[-2:], axis=1, inplace=True)
                    df.insert(df.shape[1],"Attack",[1 for i in range(df.shape[0])])
                    # calculate model accuarcy base on current dataset
                    X = df.loc[:,df.columns!="Attack"]
                    Y = df["Attack"]
                    y_pred = load_model.predict(X)
                    model_accuracy = accuracy_score(Y, y_pred)
                    # print('model_accuracy',model_accuracy)
                    return model_accuracy
        except Exception as exc:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "generateDefenceAccuracy1", exc, apiEndPoint, errorRequestMethod)
            raise Exception

    
    def checkAttackListStatus1(payload):
        try:
            statusList = []
            defenceList = []

            for filename in os.listdir(payload['folder_path']):
                if filename.endswith('.csv'):
                    csv_path = os.path.join(payload['folder_path'], filename)
                    df = pd.read_csv(csv_path)
                    col = df[df[df.columns[-1]] == True][df.columns[-1]].value_counts().tolist()
                    if len(col) > 0:
                        value = (col[0] / df.shape[0]) * 100
                        # score = Utility.generateDefenceAccuracy({'modelName':payload['modelName'], 'csv_path':csv_path})
                        score = Utility.generateDefenceAccuracy1({'modelName':payload['modelName'], 'csv_path':csv_path, 'folder_path':payload['folder_path']})
                        # statusList.append({filename.split('.')[0]:f"{value:.2f}%"})
                        # defenceList.append({filename.split('.')[0]:f"{(score*100):.2f}%"})
                        statusList.append({filename.split('.')[0]:value})
                        defenceList.append({filename.split('.')[0]:(score*100)})
                    else:
                        statusList.append({filename.split('.')[0]:0.0})
                        defenceList.append({filename.split('.')[0]:0.0})
                else:
                    if filename.endswith('.pdf') or filename.endswith('.html') or filename.endswith('.pkl'):
                        continue
                    else:
                        statusList.append({filename.split('.')[0].split('_')[1]:'True'})
                        defenceList.append({filename.split('.')[0].split('_')[1]:'Pass'})

            return statusList,defenceList
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "checkAttackListStatus1", e, apiEndPoint, errorRequestMethod)
    
    def makeAttackListRow1(payload):
        try:
            # get status message of all selected attackList
            keys_list = [key for dictionary in payload['statusList'] for key in dictionary.keys()]

            # all applicable attack arrange in list of dictionary
            attack_list_dict = []
            for i in range(len(payload['total_attacks'])):
                d = {}
                if (payload['total_attacks'][i] in payload['attackList']) and (payload['total_attacks'][i] in keys_list):
                    attack_type = ''
                    if payload['total_attacks'][i] in Utility.ArtAttackTypes['Evasion']:
                        attack_type = 'Evasion'
                    elif payload['total_attacks'][i] in Utility.ArtAttackTypes['Inference']:
                        attack_type = 'Inference'

                    val1 = ''
                    for k in payload['statusList']:
                        if payload['total_attacks'][i] == next(iter(k.keys())):
                            val1 = k[payload['total_attacks'][i]]
                            break
                    
                    val2 = ''
                    for k in payload['defenceList']:
                        if payload['total_attacks'][i] == next(iter(k.keys())):
                            val2 = k[payload['total_attacks'][i]]
                            break

                    d['name'] = payload['total_attacks'][i]
                    d['selectedAttack'] = '<i class="pass-icon">✔</i>'
                    d['status'] = val1
                    d['accuracy'] = val2
                    d['type'] = attack_type
                    attack_list_dict.append(d)
                # else:
                #     d['name'] = payload['total_attacks'][i]
                #     d['selectedAttack'] = '<i class="fail-icon">X</i>'
                #     d['status'] = "-"
                #     attack_list_dict.append(d)

            # sort this list base on Attack Type key
            attack_list_dict = sorted(attack_list_dict, key=lambda x: x['type'])

            # take all these attacks base on sorting with type key
            # attack_list = []
            # for k in attack_list_dict:
            #     attack_list.append(k['name'])
            attack_list = [{k:v for k,v in d.items() if k in ['name', 'type']} for d in attack_list_dict]

            # now all the rows write in html format
            rows = ""
            for attack in attack_list_dict:
                row = f"""
                    <tr>
                        <td>{attack['type']}</td>
                        <td>{attack['name']}</td>
                        <td>{attack['selectedAttack']}</td>
                        <td>{attack['status']}</td>
                        <td>{attack['accuracy']}</td>
                    </tr>
                """
                rows += row

            return rows, attack_list
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "makeAttackListRow1", e, apiEndPoint, errorRequestMethod)


    def makeAttackListRow(payload):
        try:
            # get status message of all selected attackList
            keys_list = [key for dictionary in payload['statusList'] for key in dictionary.keys()]

            # all applicable attack arrange in list of dictionary
            attack_list_dict = []
            for i in range(len(payload['total_attacks'])):
                d = {}
                if (payload['total_attacks'][i] in payload['attackList']) and (payload['total_attacks'][i] in keys_list):
                    attack_type = ''
                    if payload['total_attacks'][i] in Utility.ArtAttackTypes['Evasion']:
                        attack_type = 'Evasion'
                    elif payload['total_attacks'][i] in Utility.ArtAttackTypes['Inference']:
                        attack_type = 'Inference'

                    val1 = ''
                    for k in payload['statusList']:
                        if payload['total_attacks'][i] == next(iter(k.keys())):
                            val1 = k[payload['total_attacks'][i]]
                            break
                    
                    val2 = ''
                    for k in payload['defenceList']:
                        if payload['total_attacks'][i] == next(iter(k.keys())):
                            val2 = k[payload['total_attacks'][i]]
                            break

                    d['name'] = payload['total_attacks'][i]
                    d['selectedAttack'] = '<i class="pass-icon">✔</i>'
                    d['status'] = val1
                    d['accuracy'] = val2
                    d['type'] = attack_type
                    attack_list_dict.append(d)
                # else:
                #     d['name'] = payload['total_attacks'][i]
                #     d['selectedAttack'] = '<i class="fail-icon">X</i>'
                #     d['status'] = "-"
                #     attack_list_dict.append(d)

            # sort this list base on Attack Type key
            attack_list_dict = sorted(attack_list_dict, key=lambda x: x['type'])

            # take all these attacks base on sorting with type key
            # attack_list = []
            # for k in attack_list_dict:
            #     attack_list.append(k['name'])
            attack_list = [{k:v for k,v in d.items() if k in ['name', 'type']} for d in attack_list_dict]

            # now all the rows write in html format
            rows = ""
            for attack in attack_list_dict:
                row = f"""
                    <tr>
                        <td>{attack['type']}</td>
                        <td>{attack['name']}</td>
                        <td><span class="selected-attack">✔</span></td>
                        <td>
                            <div class="attack-accuracy">
                                <div class="attack-accuracy-value">{attack['status']:.2f}%</div>
                                <div class="attack-accuracy-bar">
                                    <div class="attack-accuracy-bar-fill" style="width: {attack['status']}%;"></div>
                                </div>
                            </div>
                        </td>
                        <td>
                            <div class="detection-accuracy">
                                <div class="detection-accuracy-value">{attack['accuracy']:.2f}%</div>
                                <div class="detection-accuracy-bar">
                                    <div class="detection-accuracy-bar-fill" style="width: {attack['accuracy']}%;"></div>
                                </div>
                            </div>
                        </td>
                    </tr>
                """
                rows += row

            return rows, attack_list
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "makeAttackListRow", e, apiEndPoint, errorRequestMethod)
    
    
    def createAttackFolder(payload):
        try:
            folders = list(set(d['type'] for d in payload['attack_list']))
            for folder in folders:
                if folder == 'Evasion':
                    evasion_path = os.path.join(payload['report_path'],folder)
                    if not os.path.exists(evasion_path):
                        os.mkdir(evasion_path)
                elif folder == 'Inference':
                    inference_path = os.path.join(payload['report_path'],folder)
                    if not os.path.exists(inference_path):
                        os.mkdir(inference_path)

            for filename in os.listdir(payload['report_path']):
                if filename.endswith('.csv'):
                    csv_file_path = os.path.join(payload['report_path'], filename)
                    if filename.split('.')[0] in Utility.ArtAttackTypes['Evasion']:
                        with open(csv_file_path, 'r') as csv_file:
                            with open(os.path.join(evasion_path, filename), 'w') as f:
                                shutil.copyfileobj(csv_file, f) 
                    elif filename.split('.')[0] in Utility.ArtAttackTypes['Inference']:
                        with open(csv_file_path, 'r') as csv_file:
                            with open(os.path.join(inference_path, filename), 'w') as f:
                                shutil.copyfileobj(csv_file, f)
                    Utility.databaseDelete(csv_file_path)
                elif filename.endswith('.png'):
                    png_file_path = os.path.join(payload['report_path'], filename)
                    # print(filename.split('.')[0].split('_')[1])
                    if filename.split('.')[0].split('_')[1] in Utility.ArtAttackTypes['Evasion']:
                        with open(png_file_path, 'rb') as png_file:
                            # content = png_file.read()
                            with open(os.path.join(evasion_path, filename), 'wb') as f:
                                shutil.copyfileobj(png_file, f)
                                # f.write(content) 
                    elif filename.split('.')[0].split('_')[1] in Utility.ArtAttackTypes['Inference']:
                        with open(png_file_path, 'rb') as png_file:
                            # content = png_file.read()
                            with open(os.path.join(inference_path, filename), 'wb') as f:
                                shutil.copyfileobj(png_file, f)
                                # f.write(content) 
                    Utility.databaseDelete(png_file_path)
                # elif filename.endswith('.png'):
                #     png_file_path = os.path.join(payload['report_path'], filename)
                #     if filename.split('.')[0] in Utility.ArtAttackTypes['Evasion']:
                #         with open(png_file_path, 'r', encoding='utf-8', errors='replace') as png_file:
                #             # content = png_file.read()
                #             with open(os.path.join(evasion_path, filename), 'w', encoding='utf-8') as f:
                #                 shutil.copyfileobj(png_file, f)
                #                 # f.write(content) 
                #     elif filename.split('.')[0] in Utility.ArtAttackTypes['Inference']:
                #         with open(png_file_path, 'r', encoding='utf-8', errors='replace') as png_file:
                #             # content = png_file.read()
                #             with open(os.path.join(inference_path, filename), 'w', encoding='utf-8') as f:
                #                 shutil.copyfileobj(png_file, f)
                #                 # f.write(content) 
                #     Utility.databaseDelete(png_file_path)
                # elif filename.endswith('.png'):
                #     png_file_path = os.path.join(payload['report_path'], filename)
                #     with open(png_file_path, 'r') as png_file:
                #         result = chardet.detect(png_file.read())
                #         encoding = result['encoding']
                #     if filename.split('.')[0] in Utility.ArtAttackTypes['Evasion']:
                #         with codecs.open(png_file_path, 'r', encoding=encoding, errors='ignore') as png_file:
                #             with codecs.open(os.path.join(evasion_path, filename), 'w', encoding='utf-8') as f:
                #                 shutil.copyfileobj(png_file, f)
                #     elif filename.split('.')[0] in Utility.ArtAttackTypes['Inference']:
                #         with codecs.open(png_file_path, 'r', encoding=encoding, errors='ignore') as png_file:
                #             with codecs.open(os.path.join(inference_path, filename), 'w', encoding='utf-8') as f:
                #                 shutil.copyfileobj(png_file, f)
                #     Utility.databaseDelete(png_file_path)
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "createAttackFolder", e, apiEndPoint, errorRequestMethod)

    
    def graphForAttack(payload):
        try:
            csv_path = os.path.join(payload['folder_path'],'Attack_Samples.csv')
            graph_path = os.path.join(payload['folder_path'], 'graph.png')

            # taking counts of False and True and then generate pie chart and save graph in png
            df = pd.read_csv(csv_path)
            if payload["attackName"] in Utility.ArtAttackTypes['Evasion']:
                comparasion_counts = df[[payload['target'],'prediction']].apply(
                                        lambda x: 'Successfull Data' if x[payload['target']] != x['prediction'] 
                                            else 'Unsuccessfull Data', axis=1).value_counts().to_dict()
            elif payload["attackName"] in Utility.ArtAttackTypes['Inference']:
                comparasion_counts = df[[payload['target'],'prediction']].apply(
                                        lambda x: 'Successfull Data' if x[payload['target']] == x['prediction'] 
                                            else 'Unsuccessfull Data', axis=1).value_counts().to_dict()
            plt.figure(figsize=(8,6))
            if len(comparasion_counts) > 1:
                comparasion_counts = {'Successfull Data': comparasion_counts['Successfull Data'], 
                                    'Unsuccessfull Data':comparasion_counts['Unsuccessfull Data']}
                # plt.pie(list(comparasion_counts.values()), labels=list(comparasion_counts.keys()), 
                #         colors=['#bf1029','#056517'], explode=(0.1,0), autopct='%1.1f%%', startangle=90)
                plt.pie(list(comparasion_counts.values()), labels=list(comparasion_counts.keys()), 
                        colors=['#1ca0f2','#05050F'], explode=(0.1,0), autopct='%1.1f%%', startangle=90)
            else:
                if 'Unsuccessfull Data' in comparasion_counts and 'Successfull Data' not in comparasion_counts:
                    # plt.pie(list(comparasion_counts.values()), labels=list(comparasion_counts.keys()), 
                    #     colors=['#056517'], startangle=90)
                    plt.pie(list(comparasion_counts.values()), labels=list(comparasion_counts.keys()), 
                        colors=['#05050F'], startangle=90)
                elif 'Successfull Data' in comparasion_counts and 'Unsuccessfull Data' not in comparasion_counts:
                    # plt.pie(list(comparasion_counts.values()), labels=list(comparasion_counts.keys()), 
                    #     colors=['#bf1029'], startangle=90)
                    plt.pie(list(comparasion_counts.values()), labels=list(comparasion_counts.keys()), 
                        colors=['#1ca0f2'], startangle=90)
            # cols = comparasion_counts.index.to_list()
            # plt.figure(figsize=(8,6))
            # # plt.bar(comparasion_counts.index, comparasion_counts.values, label=['True','False'])
            # if len(cols) > 1:
            #     plt.pie(comparasion_counts.values, labels=['Unsuccessfull Data','Successfull Data'], colors=['#056517','#bf1029'], explode=(0.1,0), autopct='%1.1f%%', startangle=90)
            # else:
            #     if cols[0] == 'Successfull Data':
            #         plt.pie(comparasion_counts.values, labels=['Successfull Data'], colors=['#bf1029'], autopct='%1.1f%%', startangle=90)
            #     else:
            #         plt.pie(comparasion_counts.values, labels=['Unsuccessfull Data'], colors=['#056517'], autopct='%1.1f%%', startangle=90)
            plt.legend(loc='center right', bbox_to_anchor=(1.2, 0.9))
            plt.grid(True)
            plt.savefig(graph_path)

            # saved graph.png convert into html
            with open(graph_path, 'rb') as img_file:
                img_data = img_file.read()
                base64_img = base64.b64encode(img_data).decode('utf-8')

            graph_html = f"""<div class='graph-container-attack'><img src='data:image/png;base64,{base64_img}' alt='Attack Graph' class='graph-image'></div>"""
            os.remove(graph_path)

            return graph_html
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "graphForAttack", e, apiEndPoint, errorRequestMethod)

    
    def graphForCombineAttack(payload):
        try:
            # checking each csv file like it contains target and prediction column or not or only html file
            key = 0
            files = list(os.listdir(payload['folder_path']))
            for filename in os.listdir(payload['folder_path']):
                if filename.endswith('.csv'):
                    csv_path = os.path.join(payload['folder_path'], filename)
                    df = pd.read_csv(csv_path)
                    cols = list(df.columns)
                    if payload['target'] in cols and 'prediction' in cols:
                        continue
                    else:
                        key = 1
                        break
            
            # taking counts of False and True and then generate stack bar chart and save graph in png
            html_path = os.path.join(payload['folder_path'], 'report.html')
            graph_html:any
            if key == 1 or len(files) == 1 or payload['model_metaData']['targetDataType'] == 'Image':
                graph_html = Utility.htmlContent({'modelName':payload['modelName'], 'model_metaData':payload['model_metaData'], 'reportTime':payload['reportTime'], 'success_skipped':payload['success_skipped'], 'rows':payload['rows']})
            else:
                attackList = [x['name'] for x in payload['attack_list']] 
                graph_path = os.path.join(payload['folder_path'], 'graph.png')
                df_size = []
                data = []
                for attackName in attackList:
                    for filename in os.listdir(payload['folder_path']):
                        if attackName == filename.split('.')[0]:
                            if filename.endswith('.csv'):
                                csv_path = os.path.join(payload['folder_path'], filename)
                                df = pd.read_csv(csv_path)
                                df_size.append(df.shape[0])
                                equal = len(df[df[payload['target']] == df['prediction']])
                                unequal = len(df[df[payload['target']] != df['prediction']])
                                if filename.split('.')[0] in Utility.ArtAttackTypes['Evasion']:
                                    data.append((filename.split('.')[0],equal,unequal))
                                elif filename.split('.')[0] in Utility.ArtAttackTypes['Inference']:
                                    data.append((filename.split('.')[0],unequal,equal))

                result_df = pd.DataFrame(data, columns=['','Unsuccessfull Data','Successfull Data'])
                result_df.set_index('')[['Unsuccessfull Data','Successfull Data']].plot(kind='bar', 
                                                    stacked=True, figsize=(12,8), color=['#056517','#bf1029'])
                # plt.title('Evasion Attack Robustness', color='white', bbox=dict(facecolor='darkorchid'))
                plt.ylabel('Adversial Sample Size')
                plt.legend(title='Status', bbox_to_anchor=(1.05,1), loc='upper left')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                for index, row in result_df.iterrows():
                    plt.text(index, row['Unsuccessfull Data']/2, f"{((row['Unsuccessfull Data']/df_size[index])*100):.1f}%", ha='center', va='center', color='white')
                    plt.text(index, row['Unsuccessfull Data']+row['Successfull Data']/2, f"{((row['Successfull Data']/df_size[index])*100):.1f}%", ha='center', va='center', color='white')
                plt.savefig(graph_path)

                # saved graph.png convert into html
                with open(graph_path, 'rb') as img_file:
                    img_data = img_file.read()
                    base64_img = base64.b64encode(img_data).decode('utf-8')

                graph_html = Utility.htmlContent({'modelName':payload['modelName'], 'model_metaData':payload['model_metaData'], 'reportTime':payload['reportTime'], 'success_skipped':payload['success_skipped'], 'rows':payload['rows'], 'graph':base64_img})
                os.remove(graph_path)

            # read content of combined html and add graph html content at begining
            with open(html_path, 'r') as old_html_file:
                old_html_content = old_html_file.read()
            new_html_content = graph_html + old_html_content

            # now after adding graph+combined html content then again write into same previous combined.html
            with open(html_path, 'w', encoding='utf-8') as combine_html_file:
                combine_html_file.writelines(new_html_content)
                combine_html_file.writelines(Utility.htmlCssContent())
        
            return "Success"
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "graphForCombineAttack", e, apiEndPoint, errorRequestMethod)
    

    def getPredictionsFromEndpoint(payload):

        try:
            
            api_data_variable = payload['data']
            api_response_variable = payload['prediction']

            if payload['batch']:
                # Send a post request to the end point with multiple data points
                headers ={'Content-Type': 'application/json'}
                request_data =json.dumps({api_data_variable: payload['train_data'].tolist()})
                log.info('Trying connection to model endpoint...')
                response = requests.post(payload['api'], request_data, headers=headers)
            else:
                # Send a post request to the end point with single data point
                headers ={'Content-Type': 'application/json'}
                request_data =json.dumps({api_data_variable: payload['train_data'].reshape(1, -1).tolist()})
                log.info('Trying connection to model endpoint...')
                response = requests.post(payload['api'], request_data, headers=headers)
                
            # Get the prediction from the response
            prediction = json.loads(response.text)[api_response_variable]

        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "getPredictionsFromEndpoint", e, apiEndPoint, errorRequestMethod)

        return prediction
    

    def createArtEstimator(payload):
        try:
            sklearn_api_estimator = SklearnAPIClassifier(api=payload['modelEndPoint'],
                                    nb_classes = payload['nb_classes'],
                                    input_shape = payload['input_shape'],
                                    api_data_variable = payload['api_data_variable'],
                                    api_response_variable = payload['api_response_variable']
                                )

            return sklearn_api_estimator
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "createArtEstimator", e, apiEndPoint, errorRequestMethod)

    def generateImage(payload):

        try:
            # Create a composite image with the original and adversarial images
            composite_image = np.concatenate((payload['base_sample'][0], np.ones((payload['base_sample'].shape[1], 10, 3)), payload['adversial_sample'][0]), axis=1)

            # Set the figure size and create a subplot
            fig, ax = plt.subplots(figsize=(10, 4))

            # Display the composite image
            ax.imshow(composite_image)
            ax.axis('off')

            # Set the titles for the original and adversarial images
            ax.text(0.25, -0.1, 'Original Image', transform=ax.transAxes, ha='center')
            ax.text(0.75, -0.1, 'Adversarial Image', transform=ax.transAxes, ha='center')

            # Save the composite image to a file
            img_path = os.path.join(payload['report_path'], f"{payload['attackName']}.png")
            plt.savefig(img_path)

            # Show the plot
            # plt.show()
            plt.close()

            return 
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "generateImage", e, apiEndPoint, errorRequestMethod)
    

    def getcurrentDirectory():
        try:
            root_path = os.getcwd()
            directories = root_path.split(os.path.sep)
            srcFlag = False

            for i in range(len(directories)-1, -1, -1):
                if directories[i] == 'src':
                    srcFlag = True
                    break
            if srcFlag == True:
                src_index = directories.index("src")
                new_path = os.path.sep.join(directories[:src_index])
            else:
                new_path = os.getcwd()

            return new_path
        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "getcurrentDirectory", e, apiEndPoint, errorRequestMethod)


    def isContentSafe(payload):
        try:
            for key, value in payload.items():
                if(type(value) != str):
                    return False
                for char in value:
                    if(char!='-' and char!='_' and char!=' ' and (char.isalnum()==False)):
                        return False
            return True
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "isContentSafe", e, apiEndPoint, errorRequestMethod)
    
    