'''
MIT license https://opensource.org/licenses/MIT
Copyright 2024 Infosys Ltd
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

#from gridfs import GridFS
import pymongo
from dao.databaseConnection import DB
import datetime,time
import os,zipfile,io
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


class ExternalResults:

    mycol = mydb["ExternalResults"]
    #fs = GridFS(mydb)


    def get_all(payload):

        pass
        # models = Dataset.mycol.distinct(payload)

        # return models


    def findOne(query):
        try:
            values=None
            try:
                values = ExternalResults.mycol.find(query,{})
                #print(values)
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
    

    def findallRobustnessScore(query):
        
        try:
            resultList=[]
            results=ExternalResults.mycol.find(query,{'_id':0})
            # results=ExternalResults.mycol.distinct("CoLA")
            for result in results :
                resultList.append(result)
            return resultList
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "findallRobustnessScore", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
    

    def findallAttackScore(query):
        
        try:
            resultList=[]
            results=ExternalResults.mycol.find(query,{'_id':0})
            # results=ExternalResults.mycol.distinct("TextBugger")
            for result in results :
                resultList.append(result)
            return resultList
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "findallAttackScore", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
    

    def externalRobustnessScoreCreate(values):

        try:
            value = AttributeDict(values)
            #print(value)
            print(value.modelName)
            mydoc = {
            "modelName":value.modelName,
            "SST_2":value.SST_2,
            "CoLA":value.CoLA,
            "QQP":value.QQP,
            "MPRC": value.MPRC,
            "MNLI":value.MNLI,
            "QNLI":value.QNLI,
            "RTE":value.RTE,
            "WNLI":value.WNLI,
            "MMLU":value.MMLU,
            "SQuAD_v2":value.SQuAD_v2,
            "IWSLT":value.IWSLT,
            "UN_Multi":value.UN_Multi,
            "Math":value.Math,
            "Avg":value.Avg,
            "inhouse_model":value.inhouse_model
        }
            #print(mydoc)
            resultCreatedData = ExternalResults.mycol.insert_one(mydoc)
            #print(resultCreatedData.inserted_id)
            return resultCreatedData.inserted_id
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "externalRobustnessScoreCreate", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
    

    def externalAttackScoreCreate(values):

        try:
            value = AttributeDict(values)
            mydoc = {
            "modelName":value.modelName,
            "TextBugger":value.TextBugger,
            "DeepWordBug":value.DeepWordBug,
            "TextFoller":value.TextFoller,
            "BertAttack": value.BertAttack,
            "CheckList":value.CheckList,
            "StressTest":value.StressTest,
            "Semantic":value.Semantic,
            "inhouse_model":value.inhouse_model
        }
            #print(mydoc)
            resultCreatedData = ExternalResults.mycol.insert_one(mydoc)
            #print(resultCreatedData.inserted_id)
            return resultCreatedData.inserted_id
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "externalAttackScoreCreate", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)
    


    def update(id,value:dict):

        pass
        # newvalues = { "$set": value }
        # modelUpdatedData = Model.mycol.update_one({"_id":id},newvalues)
        # log.debug(str(newvalues)) 

        # return modelUpdatedData.acknowledged
    

    def delete(query):

        try:
            ExternalResults.mycol.delete_many(query)
        except Exception as e:
            log.info(e)
            if(telemetry_flg == 'True'):
                tb = traceback.format_exc()
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "deleteExternalResults", f"traceback: {tb}\nexception: {e}", apiEndPoint, errorRequestMethod)

    
