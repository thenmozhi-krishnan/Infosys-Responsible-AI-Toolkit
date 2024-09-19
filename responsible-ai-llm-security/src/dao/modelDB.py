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
import sys
import shutil
import concurrent.futures as con
from config.logger import CustomLogger
import traceback

log =CustomLogger()

telemetry_flg =os.getenv("TELEMETRY_FLAG")

apiEndPoint ='/v1/infosys/llm/security/docs'
errorRequestMethod = 'GET'
# from batch_processing.dao.DatabaseConnection import DB
# from batch_processing.dao.DocPageDtl import *
# from batch_processing.dao.DocProcDtlDb import DocProcDtl
# from dotenv import load_dotenv
# from batch_processing.config.logger import CustomLogger
# from app.config.logger import CustomLogger
# from app.dao.DatabaseConnection import DB

# load_dotenv()
# log = CustomLogger()

mydb = DB.connect()

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Model:

    mycol = mydb["InfosysModels"]
    fs = GridFS(mydb)


    def get_all(payload):

        pass
        # models = Model.mycol.distinct(payload)

        # return models


    def findOne(query):

        try:
            values=None
            try:
                values = Model.mycol.find(query,{})[0]
                values = AttributeDict(values)
            except:
                log.info("No such item")

            return values
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "findOneExternalResults", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
    

    def findall(query):
        
        try:
            modelList=[]
            models=Model.mycol.find(query,{})
            for model in models :
                modelList.append(model)
            return modelList
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "findallExternalResults", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
    

    def create(modelDetails):

        try:
            mydoc = {
            "_id":modelDetails.modelName,
            "id":modelDetails.modelName,
            "modelName":modelDetails.modelName,   #D
            "modelDeploymentEndPoint":modelDetails.modelDeploymentEndPoint,
            "headers":modelDetails.headers,
            "payload":modelDetails.payload,
            "lastUpdatedDateTime": datetime.datetime.now(),
            }
            modelCreatedData = Model.mycol.insert_one(mydoc)
            #print(modelCreatedData.inserted_id)

            return modelCreatedData.inserted_id
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "createExternalResults", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
    

    def update(id,value:dict):

        pass
        # newvalues = { "$set": value }
        # modelUpdatedData = Model.mycol.update_one({"_id":id},newvalues)
        # log.debug(str(newvalues)) 

        # return modelUpdatedData.acknowledged
    
    def deleteFiles(modelId):
        
        try:
            Model.fs.delete(modelId)
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "deleteFilesExternalResults", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)

    def delete(query):
        
        try:
            Model.mycol.delete_many(query)
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "deleteExternalResults", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)

    def findModelDetails(modelName):
        
        try:
            modelDetails=Model.findOne({"modelName":modelName})
            return modelDetails
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "findModelDetails", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)

