'''
MIT license https://opensource.org/licenses/MIT
Copyright 2024 Infosys Ltd
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
from functools import lru_cache
import requests
import promptbench.promptbench as pb
from promptbench.promptbench.models import LLMModel
from dao.modelDB import Model as M
from dao.datasetDB import Dataset as D
from dao.databaseConnection import DB
from dao.resultsDB import Results as R
from dao.externalResultsDB import ExternalResults
import os
from gridfs import GridFS
#from dao.DatabaseConnection import DB
#from promptbench.promptbench.prompt_attack import Attack
import sys
import shutil
import concurrent.futures as con
from config.logger import CustomLogger
import traceback

log =CustomLogger()

telemetry_flg =os.getenv("TELEMETRY_FLAG")

apiEndPoint ='/v1/infosys/llm/security/docs'
errorRequestMethod = 'GET'

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class LLMs:

    def addModel(payload):
        
        try:
            value = AttributeDict(payload.modelDetails)
            modelExists=M.findOne({"modelName":value.modelName})
            if modelExists==None:
                M.create(value)
                return "Model added successfully"
            else:
                return "Model already exists"
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "addModel", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
            return "Model Addition Failed! Please Try Again"

    def modelDelete(payload):

        try:
            modelExists=M.findOne({"modelName":payload.modelName})
            if modelExists!=None:
                M.delete({"modelName":payload.modelName})
                return "Model deleted successfully"
            else:
                return "Model does not exits"
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "modelDelete", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
            return "Model Deletion Failed! Please Try Again" 

    
    def getModels():

        try:
            availableModels=M.findall({})
            modelList=[]
            for models in availableModels:
                modelList.append({"modelName":models["modelName"],"lastUpdatedDateTime":models["lastUpdatedDateTime"]})
            log.info(modelList)
            return modelList
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "getModels", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
            return "Something Went Wrong while fetching models"
        
    def addDataset(payload):
        
        try:
            datasetExists=D.findOne({"datasetName":payload.datasetName})
            if datasetExists==None:
                D.create(payload)
                return "Dataset added successfully"
            else:
                return "dataset already exists"
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "addDataset", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
            return "Data Addition Failed! Please Try Again" 
    
    def getDatasets():

        try:
            availableDatasets=D.findall({})
            datasetList=[]
            for datasets in availableDatasets:
                datasetList.append({"DatasetName":datasets["datasetName"],"LastUpdatedDateTime":datasets["lastUpdatedDateTime"]})
            log.info(datasetList)
            return datasetList
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "getDatasets", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
            return "Something Went Wrong while fetching datasets"
            
    @lru_cache(maxsize=None)
    def loadDataset(datasetName):
        
        try:
            dataPath=D.findDataFile(datasetName)
            log.info(dataPath)
            dataset = pb.DatasetLoader.load_dataset(datasetName,dataPath)
            #deleting datafile
            try:
                    shutil.rmtree(dataPath)
            except Exception as e:
                    log.info(f"An error occured: {e}")
            return dataset
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "loadDataset", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
            return "Something Went Wrong while loading dataset"
            
    def getDatasetsGlimpse(payload):
        
        try:
            #dataset=LLMs.loadDataset(payload.datasetName)
            #return dataset[:payload.numberOfEntries]
            dataset=LLMs.loadDataset(payload.datasetName)
            dict={}
            dict[payload.datasetName]=dataset[:payload.numberOfEntries]
            dict["length"]=len(dataset)
            return dict
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "getDatasetsGlimpse", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
            return "Something Went Wrong in loading dataset"
        
    def datasetDelete(payload):
        
        try:
            datasetExists=D.findOne({"datasetName":payload.datasetName}) 
            if datasetExists !=None:
                D.deleteFiles(datasetExists.datasetId)
                D.delete({"datasetName":payload.datasetName})
                return "Dataset deleted successfully"
            else:
                return "Dataset does not exits"
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "datasetDelete", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
            return "Something Went Wrong while deleting dataset"
        
    @lru_cache(maxsize=None)
    def loadModel(modelName):
        
        try:
            modelDetails=M.findModelDetails(modelName)
            log.info(modelDetails)
            return modelDetails
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "loadModel", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
            return "Something Went Wrong while loading models" 
        
    def getRobustness(payload):

        try:
            modelDetails=LLMs.loadModel(payload.modelName)
            headers = modelDetails.headers
            modelDeploymentEndPoint=modelDetails.modelDeploymentEndPoint
            dataset=LLMs.loadDataset(payload.dataset)


            #Prompt API supports a list, so you can pass multiple prompts at once.
            prompts = pb.Prompt(payload.prompts)
            #print(prompts)
            
            def proj_func(pred):
                mapping = {
                    "positive": 1,
                    "negative": 0
                }
                return mapping.get(pred, -1)
            def processRawPrediction(pred,payload):
                    if payload.dataset == "sst2":
                            if isinstance(pred,str):
                                    if "positive" in pred.lower():
                                            return 1
                                    elif "negative" in pred.lower():
                                            return 0
                                    else:
                                            return -1
                            if isinstance(pred,(int,float)):
                                    return 1 if pred >0 else 0
                    if payload.dataset == "cola":
                            if isinstance(pred,str):
                                    if "correct" in pred.lower() or "acceptable" in pred.lower()  or "right" in pred.lower():
                                            return 1
                                    elif "incorrect" in pred.lower() or "unacceptable" in pred.lower()  or "wrong" in pred.lower():
                                            return 0
                                    else:
                                            return -1
                            if isinstance(pred,(int,float)):
                                    return 1 if pred >0 else 0
                    if payload.dataset == "qqp":
                            if isinstance(pred,str):
                                    if "yes" in pred.lower()  or "equivalent" in pred.lower():
                                            return 1
                                    elif "no" in pred.lower() or "not equivalent" in pred.lower():
                                            return 0
                                    else:
                                            return -1
                            if isinstance(pred,(int,float)):
                                    return 1 if pred >0 else 0
            
            #print(len(dataset))
            # if len(dataset)>1000:
            #         dataset=dataset[:1000]
            # dataset=dataset[:10]
            dataset=dataset[:payload.numberOfSamples]
            ansDict={}
            #promptList=[]
            from tqdm import tqdm
            for prompt in prompts:
                #print(prompt)
                preds = []
                labels = []
                for data in tqdm(dataset):
                    # process input
                    input_text = pb.InputProcess.basic_format(prompt, data)
                    label = data['label']
                    # raw_pred = model(input_text)
                    modelPayload =modelDetails.payload
                    modelPayload["inputs"]=input_text
                    # modelPayload["parameters"]["max_new_tokens"]=10
                    response = requests.post(modelDeploymentEndPoint, headers=headers, json=modelPayload, verify=False)
                    if response.status_code == 200:
                            raw_pred = response.json()
                            #print("Prediction:", raw_pred)
                            # pred=processRawPrediction(raw_pred,payload)
                            pred = pb.OutputProcess.cls(raw_pred[0]['generated_text'], proj_func)
                            # pred = pb.OutputProcess.cls(raw_pred['generated_text'], proj_func)
                            preds.append(pred)
                            labels.append(label)
                    else:
                            log.info("Prediction failed with status code:", response.status_code)
                            log.info("Error message:", response.text)
                    # process output
                    #pred = pb.OutputProcess.cls(raw_pred, proj_func)
                    # pred=processRawPrediction(raw_pred)
                    # preds.append(pred)
                    # labels.append(label)
            
                # evaluate
                score = pb.Eval.compute_cls_accuracy(preds, labels)
                #print(f"{score:.3f}, {prompt}")
                ansDict[prompt]=f"{score:.3f}"
                #print(raw_pred)
                
            

                #for  storing results in DB
                #promptList.append(prompt)
            ansDict["dataset"]=payload.dataset
            ansDict["model"]=payload.modelName
            ansDict["prompts"]=payload.prompts
            ansDict["datasetLength"]=len(dataset)
            R.create(ansDict)
            # print(promptList)
            #print(ansDict)

            #deleting model file
            # shutil.rmtree(modelPath)
            # #deleting dataPath
            # shutil.rmtree(dataPath)
            return ansDict
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "getRobustness", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
            return "Something Went Wrong while calculating robustness"
        
    def getLeaderboard():
        
        try:
            query={}
            formattedResult={}
            result=R.findall(query)
            for value in result:
                    model=value["model"]
                    dataset=value["dataset"]
                    accuracy=value["accuracy"]
                    prompt=value["prompt"]
                    if prompt == "Determine the emotion of the following sentence as positive or negative: {content}" or prompt == "Can these two statements be considered equal in meaning?: {content}" or prompt == "Assess the following sentence and determine if it is grammatically correct: {content}":
                            if model not in formattedResult:
                                    formattedResult[model]={}
                            formattedResult[model][dataset]=accuracy
            return formattedResult
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "getLeaderboard", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
            return "Something Went Wrong while loading leaderboard"
        
    def results(payload):
        # if R.findOne(payload.resultId):
        pass

    def addExternalRobustnessScore(payload):
        
        try:
            key = ExternalResults.externalRobustnessScoreCreate(payload.RobustnessScore)
            if key!=None:
                    return "Score added successfully."
            else:
                    return "Score not added due to some error please try again"
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "addExternalRobustnessScore", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
            return "Something Went Wrong while sending data to database" 
    
    def addExternalAttackScore(payload):
        
        try:
            key=ExternalResults.externalAttackScoreCreate(payload.attackScore)
            if key!=None:
                    return "Score added successfully."
            else:
                    return "Score not added due to some error please try again"
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "addExternalAttackScore", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
            return "Something Went Wrong while sending data to database" 

    def getExternalRobustnessScore():
        
        try:
            result=ExternalResults.findallRobustnessScore({"CoLA": {"$exists" : True}})
            return result
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "getExternalRobustnessScore", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
            return "Something Went Wrong while fetching data from database" 
        
    def getExternalAttackScore():
        
        try:
            result=ExternalResults.findallAttackScore({"TextBugger": {"$exists" : True}})
            return result
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "getExternalAttackScore", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
            return "Something Went Wrong while fetching data from database" 
        
    def deleteExternalRobustnessScore(payload):
        
        try:
            ExternalResults.delete(payload.RobustnessScore)
            return "Score deleted successfully"
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "deleteExternalRobustnessScore", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
            return "Something Went Wrong while deleting score"
        
    def deleteExternalAttackScore(payload):
        
        try:
            ExternalResults.delete(payload.attackScore)
            return "Score deleted successfully"
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "deleteExternalAttackScore", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
            return "Something Went Wrong while deleting score"


    
            

        
