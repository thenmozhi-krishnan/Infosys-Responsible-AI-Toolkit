'''
MIT license https://opensource.org/licenses/MIT
Copyright 2024 Infosys Ltd
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''


from gridfs import GridFS
import pymongo
from dao.databaseConnection import DB
import datetime,time
import os,zipfile,io
import shutil
import sys
import requests
from dotenv import load_dotenv
import concurrent.futures as con
from config.logger import CustomLogger
import traceback

log =CustomLogger()

apiEndPoint ='/v1/infosys/llm/security/docs'
errorRequestMethod = 'GET'
# from batch_processing.dao.DatabaseConnection import DB
# from batch_processing.dao.DocPageDtl import *
# from batch_processing.dao.DocProcDtlDb import DocProcDtl
# from dotenv import load_dotenv
# from batch_processing.config.logger import CustomLogger
# from app.config.logger import CustomLogger
# from app.dao.DatabaseConnection import DB

load_dotenv()
# log = CustomLogger()

mydb = DB.connect()
db_type =os.getenv("DB_TYPE")
dataset_container =os.getenv("DATASETCONTAINERNAME")
add_file =os.getenv("ADDFILEURL")
delete_file=os.getenv("DELETEFILEURL")
fetch_file =os.getenv("GETFILEURL")
telemetry_flg =os.getenv("TELEMETRY_FLAG")


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Dataset:

    mycol = mydb["Datasets"]
    fs = GridFS(mydb)


    def get_all(payload):

        pass
        # models = Dataset.mycol.distinct(payload)

        # return models


    def findOne(query):

        try:
            values=None
            try:
                values = Dataset.mycol.find(query,{})[0]
                values = AttributeDict(values)
            except:
                log.info("No such item")

            return values
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "findOneDataset", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
    

    def findall(query):
        
        try:
            datasetList=[]
            datasets=Dataset.mycol.find(query,{})
            for dataset in datasets :
                datasetList.append(dataset)
            return datasetList
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "findallDataset", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
           
    

    def create(payload):

        try:
            value = AttributeDict(payload)
            if db_type =="mongo":
                
                datasetId=Dataset.fs.put(payload.datasetFile.file.read(),filename=value.datasetName)
                localTime = time.time()
                time.sleep(1/1000)
                mydoc = {
                "_id":value.datasetName,
                "id":value.datasetName,
                "datasetId":datasetId,
                "datasetName":value.datasetName,   #D
                #"createdDateTime": datetime.datetime.now(),
                "lastUpdatedDateTime": datetime.datetime.now(),
                }
            else:
                #print("file:",payload.datasetFile.file)
                # with open(payload.datasetFile.file.read(), 'rb') as file:
                #     print("%^&&")
                response =requests.post(url =add_file, files ={"file":payload.datasetFile.file}, data ={"container_name":dataset_container})
                #print(response.status_code)
                blob_name =response.json()["blob_name"]
                localTime = time.time()
                time.sleep(1/1000)
                mydoc = {
                "_id":value.datasetName,
                "id":value.datasetName,
                "datasetId":blob_name,
                "datasetName":value.datasetName,   #D
                #"createdDateTime": datetime.datetime.now(),
                "lastUpdatedDateTime": datetime.datetime.now(),
                }
            datasetCreatedData = Dataset.mycol.insert_one(mydoc)
            #print(datasetCreatedData.inserted_id)

            return datasetCreatedData.inserted_id
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "createDataset", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
    

    def update(id,value:dict):

        pass
        # newvalues = { "$set": value }
        # modelUpdatedData = Model.mycol.update_one({"_id":id},newvalues)
        # log.debug(str(newvalues)) 

        # return modelUpdatedData.acknowledged
    
    def deleteFiles(datasetId):
        
        try:
            if db_type=="mongo":
            
                Dataset.fs.delete(datasetId)
            else:
                requests.post(url =delete_file, data ={"blob_name":datasetId,"container_name":dataset_container})
            #print("dataset deleted successfully")
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "deleteFilesDataset", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)

    def delete(query):
        
        try:
            Dataset.mycol.delete_many(query)
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "deleteDataset", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
        

    def findDataFile(datasetName):
        
        try:
            #print(datasetName)
            datasetDetails=Dataset.findOne({"datasetName":datasetName})
            if db_type=="mongo":
                dataFile=Dataset.fs.find_one({"_id":datasetDetails.datasetId})
                if dataFile:
                    data=dataFile.read()
                    try:
                        with zipfile.ZipFile(io.BytesIO(data),'r') as z:
                            z.extractall(os.path.join(os.getcwd(),f"{datasetName}"))
                    except Exception as e:
                        log.info(e)
                    return os.path.join(os.getcwd(),f"{datasetName}")
                else:
                    log.info("dataset not retrived")
                    
            else:
                dataFile =requests.get(url =fetch_file, params ={"blob_name":datasetDetails.datasetId,"container_name":dataset_container})
                binary_data =dataFile.content
                temp = io.BytesIO(binary_data)
            # print("type of temp:",type(temp))
                if temp:
                    data =temp.read()
                    try:
                        with zipfile.ZipFile(io.BytesIO(data),'r') as z:
                            z.extractall(os.path.join(os.getcwd(),f"{datasetName}"))
                    except Exception as e:
                        log.info(e)
                    return os.path.join(os.getcwd(),f"{datasetName}")
                
                else:
                    log.info("dataset not retrieved")
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "findDataFile", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
        
