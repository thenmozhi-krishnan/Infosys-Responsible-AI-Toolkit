import os
import pymongo
from dotenv import load_dotenv
from modeldeployment.config.logger import CustomLogger,request_id_var
import sys
import json
import requests
import urllib.parse
import requests
import json
import traceback

log = CustomLogger()
load_dotenv()
class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


myclient = pymongo.MongoClient(os.getenv("MONGO_PATH"))
print(myclient)
# dbname = os.getenv("APP_MONGO_DBNAME")
dbname = os.getenv("DB_NAME")

class DB:
    def connect():
        try:
            # myclient = pymongo.MongoClient(os.getenv("MONGO_PATH")) 
            # mydb = myclient[os.getenv("DB_NAME")]
            mydb = myclient[dbname]
            
            return mydb
        except Exception as e:
            log.error("error in DB connection")
            log.error(str(e))
            sys.exit()

mydb=DB.connect()
class ProfaneWords:
    mycol = mydb["ProfaneWords"]
    def findOne(id):
        try:
            values=ProfaneWords.mycol.find({"_id":id},{})[0]
            # print(values)
            values=AttributeDict(values)
            return values
        except Exception as e:
            log.error("Error occured in ProfaneWords")
            log.error(f"Exception: {e}")
    
class feedbackdb:
    feedback_collection = mydb["feedback"]
    def create(value):
        try:
            PtrnRecogCreatedData = feedbackdb.feedback_collection.insert_one(value)
            print("PtrnRecogCreatedData.acknowledged",PtrnRecogCreatedData.acknowledged)
            return PtrnRecogCreatedData.acknowledged
        except Exception as e:
            log.error("Error occured in feedbackdb")
            log.error(f"Exception: {e}")

class Results:
    mycol = mydb["moderationtelemetrydata"]
    logdb=mydb["Logdb"]
    # mycol = mydb["Results"]
    mycol2 = mydb["Results"]
    # mycol2 = mydb["Resultswithfeedback"]
    def findOne(id):
        try:
            values=Results.mycol.find({"_id":id},{})[0]
            # print(values)
            values=AttributeDict(values)
            return values
        except Exception as e:
            log.error("Error occured in Results findOne")
            log.error(f"Exception: {e}")
    def findall(query):
        try:
            value_list=[]
            values=Results.mycol.find(query,{})
            for v in values:

                v=AttributeDict(v)
                value_list.append(v)
            return value_list
        except Exception as e:
            log.error("Error occured in Results findall")
            log.error(f"Exception: {e}")

    def create(value,id,portfolio, accountname):
        request_id_var.set(id)
        try:
            value=json.loads(value.json())
            id=value["uniqueid"]
            print(id)
            mydoc={"_id":id , "created":value["created"],"portfolio":portfolio,"accountname":accountname,
                "Moderations":value["moderationResults"]}
            PtrnRecogCreatedData = Results.mycol.insert_one(mydoc)
            print("PtrnRecogCreatedData.acknowledged",PtrnRecogCreatedData.acknowledged)
            return PtrnRecogCreatedData.acknowledged
        except Exception as e:
            log.error("Error occured in Results create")
            log.error(f"Exception: {e}")

    def createlog(value):
        
        try:
            print(value)
            PtrnRecogCreatedData = Results.logdb.insert_one(value)
            print("PtrnRecogCreatedData.acknowledged",PtrnRecogCreatedData.acknowledged)
            return PtrnRecogCreatedData.acknowledged
        except Exception as e:
            log.error("Error occured in Results create")
            log.error(f"Exception: {e}")
    
    
    def createwithfeedback(value):
        
        try:
            # print(id)
            PtrnRecogCreatedData = Results.mycol2.insert_one(value)
            print("PtrnRecogCreatedData.acknowledged",PtrnRecogCreatedData.acknowledged)
            return PtrnRecogCreatedData.acknowledged
        except Exception as e:
            log.error("Error occured in createwithfeedback")
            log.error(f"Exception: {e}")
    
    def update(query,value:dict):
        try:
        
            newvalues = { "$set": value }
            
            PtrnRecogUpdatedData=Results.mycol.update_one(query,newvalues)
            log.debug(str(newvalues)) 
            return PtrnRecogUpdatedData.acknowledged
        except Exception as e:
            log.error("Error occured in Results update")
            log.error(f"Exception: {e}")
    
    def delete(id):
        try:
            return Results.mycol.delete_one({"_id": id})
            return Results.mycol.delete_many({"dataRecogGrpId":id}).acknowledged
            # DocProcDtl.mycol.delete_many({})
            # Docpagedtl.mycol.delete_many({})
        except Exception as e:
            log.error("Error occured in Results delete")
            log.error(f"Exception: {e}")
            
    def deleteMany(query):
        try:
            return Results.mycol.delete_many(query).acknowledged
        except Exception as e:
            log.error("Error occured in Results deleteMany")
            log.error(f"Exception: {e}")
            