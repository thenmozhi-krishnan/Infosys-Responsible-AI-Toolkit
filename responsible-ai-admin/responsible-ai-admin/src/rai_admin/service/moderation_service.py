'''
MIT license https://opensource.org/licenses/MIT
Copyright 2024 Infosys Ltd
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from io import BytesIO
import io
import json
import os
import time
from datetime import datetime



from rai_admin.dao.AuthorityDb import AuthorityDb

from rai_admin.dao.OpenAIDb import OpenAIDb

from rai_admin.dao.ConfigApiDb import ConfigApiDb
from rai_admin.dao.AccMasterDb import AccMasterDb
from rai_admin.dao.AccPtrnMappingDb import AccPtrnDb
from rai_admin.dao.AccDataGrpMappingDb import AccDataGrpDb
from rai_admin.dao.AccSafetyMappingDb import AccSafetyDb
from rai_admin.dao.ptrnRecog import PtrnRecog
from rai_admin.dao.DataRecogdb import RecogDb
from rai_admin.dao.EntityDb import EntityDb
from rai_admin.mappers.RecognizerMapper import DataEntitiesResponse, RecogStatus,RecogResponse
from rai_admin.mappers.AccMasterMapper import *
from rai_admin.mappers.PtrnRecogMapper import *
from rai_admin.mappers.PrivacyMapper import *
from rai_admin.mappers.FmConfigMapper import *

from rai_admin.mappers.SafetyMapper import *
from rai_admin.dao.FmConfigDb import FmConfigDb

from rai_admin.dao.FmConfigGrpMappingDb import FmConfigGrpDb
from rai_admin.dao.ModerationCheckDb import ModerationChecksDb,OutputModerationChecksDb
from rai_admin.dao.RestrictedtopicDetailsDb import RestrictedTopic
from rai_admin.dao.ThemeTextsDb import ThemeText
from rai_admin.dao.AdminException import ExceptionDb
from rai_admin.config.logger import request_id_var
import pandas as pd
from rai_admin.config.logger import CustomLogger
from rai_admin.dao.AccDataGrpMappingDb import AccDataGrpDb
from rai_admin.dao.customeTemplateDb import CustomeTemplateDB
from rai_admin.dao.templateDataDb import TemplatDataeDB
from rai_admin.mappers.customeTemplateMapper import *
from rai_admin.dao.MasterTemplateMappingDb import AccTemplateMap

log = CustomLogger()
class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

class ModerationService:
    def createTemplate(payload:CustomeTemplateReq)->CustomeTemplateStatus:
        try:
            log.info(f"createTemplate: {payload}")
            obj=CustomeTemplateStatus
            if(len(CustomeTemplateDB.findall({"templateName":payload.templateName,"userId":payload.userId,"mode":payload.mode}))>0):
                obj.status="Template Name already exists for user {} and mode {}".format(str(payload.userId),str(payload.mode))
                return obj
            # if payload.templateData.file:
                # payload.templateData=payload.templateData.file.read().decode("utf-8")
            # print(payload.subTemplates)
            # with open("temp.txt","w") as f:
            #     f.write(payload.templateData)
            # payload.templateData="\n".join(payload.templateData.split("\n"))
            # payload.uniqueId=payload.templateName+payload.userId
            temp_id=CustomeTemplateDB.create(payload)
            for templates in payload.subTemplates:
                templatevalue={"templateId":temp_id,"template":templates.subtemplate,"templateData":templates.templateData}
                print("Template Value",templatevalue)
                res=TemplatDataeDB.create(templatevalue)  
     
                 
            log.debug("createTemplate status: "+str(res))
            
            obj.status=str(res)
            return obj
        except Exception as e:
            log.error(str(e))
            obj=CustomeTemplateStatus
  
            obj.status="False"
            return obj
    
    def getTemplate(payload)->CustomeTemplateRes:
        try:
            log.info(f"getTemplate")
            res=CustomeTemplateDB.findall({"$or": [{"userId":payload.userId,"mode":"Private_Template"},{"mode":"Master_Template"}]})
            # print(res)
            for i in res:
                subres=TemplatDataeDB.findall({"templateId":i.templateId})
                print(subres)
                # subres=[subTemplate({t}) for t in subres]
                i.subTemplates=subres
   
            
            log.debug("Template List"+str(res))
            obj=CustomeTemplateRes(templates=res)
            # print("Template List===========",obj)
            return obj
        except Exception as e:
            log.error(str(e))
            # obj=CustomeTemplateStatus
            return e
        
        
    def getTempData(payload)->CustomeTemplateRes:
        try:
            log.info(f"getTemplate")
            res=CustomeTemplateDB.findall({"templateId":float(payload.templateId)})
            # print(res)
            for i in res:
                subres=TemplatDataeDB.findall({"templateId":i.templateId})
                print(subres)
                # subres=[subTemplate({t}) for t in subres]
                i.subTemplates=subres
   
            
            log.debug("Template List"+str(res))
            obj=CustomeTemplateRes(templates=res)
            # print("Template List===========",obj)
            return obj
        except Exception as e:
            log.error(str(e))
            # obj=CustomeTemplateStatus
            return e
    
    def updateTemplate(payload:CustomeTemplateReq)->CustomeTemplateStatus:
        try:
            log.info(f"createTemplate: {payload}")
            obj=CustomeTemplateStatus
            # if(payload.mode=="Master"):
                
                
            tempres=CustomeTemplateDB.findall({"templateName":payload.templateName,"userId":payload.userId,"mode":payload.mode})
            # print(tempres)
            if(len(tempres)==0):
                obj.status="Template dosent exists"
                return obj
            subtempres=TemplatDataeDB.findall({"templateId":tempres[0].templateId,"template":payload.subTemplates[0].subtemplate})
            # print("=====",subtempres)
            if(len(subtempres)==0):
                obj.status="Template subtemplates not exists"
                return obj
            
            res=TemplatDataeDB.update(subtempres[0]._id,{"templateData":payload.subTemplates[0].templateData})
            log.debug("UpdateTemplate status: "+str(res))
            
            obj.status=str(res)
            return obj
        except Exception as e:
            log.error(str(e))
            obj=CustomeTemplateStatus
  
            obj.status="False"
            return obj
        
    def deleteTemplate(payload)->CustomeTemplateStatus:
        try:
            log.info(f"createTemplate: {payload}")
            obj=CustomeTemplateStatus
            templist=CustomeTemplateDB.findall({"templateId":float(payload.templateId)})
            print(templist)
            if(len(templist)==0):
                obj.status="Template dosent exists"
                return obj
            # payload.templateData="\n".join(payload.templateData.split("\n"))
            res=CustomeTemplateDB.delete(float(payload.templateId))
            res=TemplatDataeDB.delete({"templateId":float(payload.templateId)})

            log.debug("DeleteTemplate status: "+str(res))
            
            obj.status=str(res)
            return obj
        except Exception as e:
            log.error(str(e))
            obj=CustomeTemplateStatus
  
            obj.status="False"
            return obj
    
    def deleteSubTemplate(payload)->CustomeTemplateStatus:
        try:
            log.info(f"createTemplate: {payload}")
            obj=CustomeTemplateStatus
            templist=TemplatDataeDB.findall({"subTemplateId":float(payload.subTemplateId)})
            print(templist)
            if(len(templist)==0):
                obj.status="Template dosent exists"
                return obj
            # payload.templateData="\n".join(payload.templateData.split("\n"))
            res=TemplatDataeDB.delete({"_id":float(payload.subTemplateId)})
            log.debug("DeleteTemplate status: "+str(res))
            
            obj.status=str(res)
            return obj
        except Exception as e:
            log.error(str(e))
            obj=CustomeTemplateStatus
  
            obj.status="False"
            return obj    
        
    def getTemplateFile(userId)->CustomeTemplateRes:
        try:
            log.info(f"getTemplate")
            res=CustomeTemplateDB.findall({"$or": [{"userId":userId,"mode":"Private_Template"},{"mode":"Master_Template"}]})
          
            f=open("template.txt","w")
            tempval=""
            for i in res:
                tempval+=i.templateName+'="""'+ i.templateData+'"""\n\n'
            f.write(tempval)
            f.close()
            log.debug("Template List"+str(res))
            
            return "template.txt"
        except Exception as e:
            log.error(str(e))
            # obj=CustomeTemplateStatus
            return e


class TempMap:     
    def AccTempMap(payload:AccTempMapReq)->CustomeTemplateStatus:
        try:
            log.info(f"AccTempMap: {payload}")
            obj=CustomeTemplateStatus
            acc=AccMasterDb.findall({"account":payload.account,"portfolio":payload.portfolio})
            if(len((acc))==0):
                obj.status="Account Name dosent exists"
                return obj
            accId=acc[0].accMasterId
            accTempMap = AccTemplateMap.findall({"accMasterId":accId})
            if(len(accTempMap)>0):
                #  obj=CustomeTemplateStatus
                 obj.status="False"
                 return obj
 
            grpValue={"accMasterId":accId,"userId":payload.userId,"requestTemplate":payload.requestTemplate,"responseTemplate":payload.responseTemplate,"comparisonTemplate":payload.comparisonTemplate}
            res=AccTemplateMap.create(grpValue)
            
            # res=MasterTemplateMap.create(payload)
            log.debug("AccTempMap status: "+str(res))
            
            obj.status=str(res)
            return obj
        except Exception as e:
            log.error(str(e))
            obj=CustomeTemplateStatus
  
            obj.status="False"
            return obj
        
    def getAccTempMap(payload)->AccTempMap:
        try:
            log.info(f"getAccTempMap")
            res=AccTemplateMap.findall({"userId":payload.userId})
            for i in res:
                subres=AccMasterDb.findall({"accMasterId":i.accMasterId})
                print(subres)
                # subres=[subTemplate({t}) for t in subres]
                i.account=subres[0].account
                i.portfolio=subres[0].portfolio

            
            log.debug("Template List"+str(res))
            obj=AccTempMap(accList=res)
            # print("Template List===========",obj)
            return obj
        except Exception as e:
            log.error(str(e))
            # obj=CustomeTemplateStatus
            return e
    
    def getTempMap(payload):
        try:
            log.info(f"getAccTempMap")
            acc=AccMasterDb.findall({"accMasterId":float(payload.accMasterId)})
            
            if(len((acc))==0):
                obj=["Account Name dosent exists"]
                return obj
            
            res=AccTemplateMap.findall({"accMasterId":float(payload.accMasterId)})
            if(len(res)>0):
                del res[0]._id
            # if()
                res[0].account=acc[0].account
                res[0].portfolio=acc[0].portfolio
            # for i in res:
            #     subres=AccMasterDb.findall({"accMasterId":i.accMasterId})
            #     print(subres)
            #     # subres=[subTemplate({t}) for t in subres]
            #     i.account=subres[0].account
            #     i.portfolio=subres[0].portfolio

            
            log.debug("Template List"+str(res))
            # obj=AccTempMap(accList=res)
            # print("Template List===========",obj)
            return res
        except Exception as e:
            log.error(str(e))
            # obj=CustomeTemplateStatus
            return
    def addTempMap(payload:AccTempMapReq)->CustomeTemplateStatus:
        try:
            log.info(f"createTemplate: {payload}")
            obj=CustomeTemplateStatus
            # acc=AccMasterDb.findall({"account":payload.account,"portfolio":payload.portfolio})
            # if(len((acc))==0):
            #     obj.status="Account Name dosent exists"
            #     return obj
            # accId=acc[0].accMasterId
            accTempMap = AccTemplateMap.findall({"mapId":float(payload.mapId),"userId":payload.userId})
            print(accTempMap)
            if(len(accTempMap)==0):
                #  obj=CustomeTemplateStatus
                 obj.status="False"
                 return obj
 
            value={"requestTemplate":accTempMap[0]["requestTemplate"]+payload.requestTemplate,"responseTemplate":accTempMap[0]["responseTemplate"]+payload.responseTemplate,"comparisonTemplate":accTempMap[0]["comparisonTemplate"]+payload.comparisonTemplate} 
            print(value)
            res=AccTemplateMap.update(float(payload.mapId),value)
            
            log.debug("AccTempMap status: "+str(res))
            
            obj.status=str(res)
            return obj
        except Exception as e:
            log.error(str(e))
            obj=CustomeTemplateStatus
  
            obj.status="False"
            return obj
    def removeTempMap(payload:RemoveTempMap)->CustomeTemplateStatus:
        try:
            log.info(f"createTemplate: {payload}")
            obj=CustomeTemplateStatus
            # acc=AccMasterDb.findall({"account":payload.account,"portfolio":payload.portfolio})
            # if(len((acc))==0):
            #     obj.status="Account Name dosent exists"
            #     return obj
            # accId=acc[0].accMasterId
            accTempMap = AccTemplateMap.findall({"mapId":float(payload.mapId),"userId":payload.userId})
            print(accTempMap)
            if(len(accTempMap)==0):
                #  obj=CustomeTemplateStatus
                 obj.status="False"
                 return obj
            print(payload.templateName)
            l=accTempMap[0][payload.tempType]
            l.remove(payload.templateName)
            
            value={payload.tempType:l} 
            print(value)
            res=AccTemplateMap.update(float(payload.mapId),value)
            # res=False
            log.debug("AccTempMap status: "+str(res))
            
            obj.status=str(res)
            return obj
        except Exception as e:
            log.error(str(e))
            obj=CustomeTemplateStatus
  
            obj.status="False"
            return obj
    def deleteTempMap(payload)->CustomeTemplateStatus:
        try:
            log.info(f"createTemplate: {payload}")
            obj=CustomeTemplateStatus
            acc=AccMasterDb.findall({"account":payload.account,"portfolio":payload.portfolio})
            if(len((acc))==0):
                obj.status="Account Name dosent exists"
                return obj
            templist=AccTemplateMap.findall({"accMasterId":acc[0].accMasterId,"userId":payload.userId})
            print(templist)
            if(len(templist)==0):
                obj.status="Template Mapping dosent exists"
                return obj
            # payload.templateData="\n".join(payload.templateData.split("\n"))
            # res=CustomeTemplateDB.delete(float(payload.templateId))
            # res=TemplatDataeDB.delete({"templateId":float(payload.templateId)})
            res=AccTemplateMap.delete({"accMasterId":acc[0].accMasterId,"userId":payload.userId})
            log.debug("DeleteTemplate status: "+str(res))
            
            obj.status=str(res)
            return obj
        except Exception as e:
            log.error(str(e))
            obj=CustomeTemplateStatus
  
            obj.status="False"
            return obj
        