'''
Copyright 2024 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import os
import traceback
import requests
import json
import datetime
from requests.auth import HTTPBasicAuth


from config.logger import CustomLogger, request_id_var

tel_env=os.getenv("TELEMETRY_ENVIRONMENT")
telemetryurl = os.getenv("TELEMETRY_PATH") 
coupledtelemetryurl=os.getenv("COUPLEDTELEMETRYPATH")
evalLLMtelemetryurl=os.getenv("EVALLLMTELEMETRYPATH")

log=CustomLogger()

class telemetry:
    # tel_flag=True
    tel_flag=os.getenv("TEL_FLAG")
    # gpt44 = os.getenv("OPENAI_API_BASE_GPT4")
    # log.info(f"TELEMETRY FLAG IN TELEMETRY CLASS-->{gpt44}")
    log.info(f"TELEMETRY FLAG IN TELEMETRY METHOD--> {tel_flag}")
    def send_coupledtelemetry_request(moderation_telemetry_request,id,portfolioName=None,accountName=None,dict_timecheck = None):
        try:
            log.info(f"Telemetry Flag inside telemetry method--> {telemetry.tel_flag}")
            # telFlagData = telemetrydb.TelemetryFlag.find({"Module":"AiCloudModeration"},{})[0]
            # tel_Falg = telFlagData["TelemetryFlag"]
            request_id_var.set(id)
            if tel_env!="IS":
                if telemetry.tel_flag=="True":
                    log.info("Inside Telemetry")
                    moderation_telemetry_request = moderation_telemetry_request.dict()
                    if portfolioName:
                        moderation_telemetry_request["portfolioName"]=portfolioName
                        moderation_telemetry_request["accountName"]=accountName
                        
                    else:
                        moderation_telemetry_request["portfolioName"]="None"
                        moderation_telemetry_request["accountName"]="None"
                    moderation_telemetry_request['Moderation layer time'] = dict_timecheck
                    # print("Inside Coupled Telemetry")
                    print("coupled data : ",json.dumps(moderation_telemetry_request))
                    response = requests.post(coupledtelemetryurl, json=moderation_telemetry_request)
                    # print("json -- ", moderation_telemetry_request.json())
                    # print("response ----- ",response)
                    # print("json ------ ",json.dumps(moderation_telemetry_request))
                    log.info(" ------------------ sending to telemetry ----------------- ")
                    response.raise_for_status()
                    log.info(" ------------------ sent to telemetry ----------------- ")
        except Exception as e:
            logobj = {"_id":id,"Telemetryerror":{"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Telemetry",
                                                   "Payload":moderation_telemetry_request,
                                                   "portfolio":portfolioName,"acount":accountName}}
            # Results.createlog(logobj)
            log.error(str(logobj))
            log.error("Error occured in Coupled Telemetry")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

    def send_telemetry_request(moderation_telemetry_request,id,lotNumber,portfolioName,accountName,userid,header=None,token_info=None,timecheck=None,modeltime=None,totaltimeforallchecks=None):
        try:

            request_id_var.set(id)
            if tel_env == "AZURE":
                log.info(f"tel_env---- >>>>>{tel_env}")
                if telemetry.tel_flag=="True":
                    
                    if token_info is None:
                        token_info = {"unique_name":"None","X-Correlation-ID":"None","X-Span-ID":"None"}
                    moderation_telemetry_request = moderation_telemetry_request.dict()
                    moderation_telemetry_request['Moderation layer time'] = {
                        "Time for each individual check": timecheck,
                        "Time taken by each model": modeltime,
                        "Total time for moderation Check": totaltimeforallchecks
                    }
                    if portfolioName:
                        
                        moderation_telemetry_request["portfolioName"]=portfolioName
                        moderation_telemetry_request["accountName"]=accountName
                        moderation_telemetry_request["lotNumber"]=lotNumber
                        moderation_telemetry_request["userid"]=userid
                        
                        response = requests.post(telemetryurl, json=moderation_telemetry_request,verify=False)
                        log.info("Inside moderation Telemetry  1 ")
                        # print("moderation_telemetry_request--->>>",json.dumps(moderation_telemetry_request))
                        response.raise_for_status()
                        log.info("--------------- Sent from moderation Telemetry  1 ----------- ")
                    else:
                        
                        moderation_telemetry_request["portfolioName"]="None"
                        moderation_telemetry_request["accountName"]="None"
                        moderation_telemetry_request["lotNumber"]="None"
                        moderation_telemetry_request["userid"]="None"

                        # print("moderation_telemetry_request--->>>",json.dumps(moderation_telemetry_request))
                        response = requests.post(telemetryurl, json=moderation_telemetry_request)
                        log.info("Inside moderation Telemetry  2")
                        response.raise_for_status()
                        log.info("--------------- Sent from moderation Telemetry  2 ----------- ")
            elif tel_env == "ETA":
                log.info(f"tel_env---- >>>>>{tel_env}")
                if telemetry.tel_flag=="True":                        

                    moderation_telemetry_request = moderation_telemetry_request.dict()
                    moderation_telemetry_request['Moderation layer time'] = {
                        "Time for each individual check": timecheck,
                        "Time taken by each model": modeltime,
                        "Total time for moderation Check": totaltimeforallchecks
                    }
                    username = os.getenv("ETA_TELEMETRY_USERNAME")
                    password = os.getenv("ETA_TELEMETRY_PASSWORD")
                    if portfolioName:
                        print("eta telemetry with portfolioname")
                        moderation_telemetry_request["portfolioName"]=portfolioName
                        moderation_telemetry_request["accountName"]=accountName
                        moderation_telemetry_request["lotNumber"]=lotNumber
                        moderation_telemetry_request["userid"]=userid
                        
                        

                        index_name = f"responsible-ai-moderation_{str(datetime.date.today()).replace('/','_')}"  # this is to make sure that the index name is unique per day and it's easy to clear the data after some day
                        doc_id = moderation_telemetry_request["uniqueid"]

                        # print("index_name",index_name)
                        url=os.getenv("ETA_TELEMETRY_ENDPOINT")
                        url = f"{url}/{index_name}/_doc/{doc_id}"
                        headers = {
                            "Content-Type": "application/json",
                        }

                        payload = {"data": moderation_telemetry_request} # this is the data that you want to send to the open search
                        #print("data --->>>",payload)
                        log.info("Inside moderation Telemetry  1 ")
                        response = requests.request(
                            "POST",
                            url,
                            headers=headers,
                            auth=HTTPBasicAuth(username,password),
                            data=json.dumps(payload),
                        )

                        if response.status_code >= 200 and response.status_code < 300:
                            print(f"Success - {response.text}")
                            log.info("--------------- Sent from moderation ETA Telemetry  1 ----------- ")
                        else:
                            print(f"error - {response.text}")
                            log.info("--------------- Error from moderation ETA Telemetry  1 ----------- ")
                        
                        
                        # response = requests.post(privacytelemetryurl, json=schema,verify=False)
                        # log.info("Inside moderation Telemetry  1 ")
                        # print("moderation_telemetry_request--->>>",json.dumps(schema))
                        # response.raise_for_status()
                        # log.info("--------------- Sent from moderation ETA Telemetry  1 ----------- ")
                    else:
                        print("eta telemetry without  portfolioname")
                        moderation_telemetry_request["portfolioName"]="None"
                        moderation_telemetry_request["accountName"]="None"
                        moderation_telemetry_request["lotNumber"]="None"
                        moderation_telemetry_request["userid"]="None"
                        
                        index_name = f"responsible-ai-moderation_{str(datetime.date.today()).replace('/','_')}"  # this is to make sure that the index name is unique per day and it easy to clear the data after some day
                        doc_id = moderation_telemetry_request["uniqueid"]#"1"  # if you dont want doc_id then remove this line and from the url also
                                                
                        url=os.getenv("ETA_TELEMETRY_ENDPOINT")
                        url = f"{url}/{index_name}/_doc/{doc_id}"

                        headers = {
                            "Content-Type": "application/json",
                        }

                        payload = {"data": moderation_telemetry_request} # this is the data that you want to send to the open search
                        #print("data --->>>",payload)
                        log.info("Inside moderation ETA Telemetry  2 ")
                        response = requests.request(
                            "POST",
                            url,
                            headers=headers,
                            auth=HTTPBasicAuth( username, password),
                            data=json.dumps(payload),
                        )

                        if response.status_code >= 200 and response.status_code < 300:
                            print(f"Success - {response.text}")
                            log.info("--------------- Sent from moderation ETA Telemetry  2 ----------- ")
                        else:
                            print(f"error - {response.text}")
                            log.info("--------------- Error from moderation ETA Telemetry  2 ----------- ")
                        
                        
                        # print("moderation_telemetry_request--->>>",json.dumps(schema))
                        # response = requests.post(privacytelemetryurl, json=schema)
                        # log.info("Inside moderation Telemetry  2")
                        # response.raise_for_status()
                        # log.info("--------------- Sent from moderation Telemetry  2 ----------- ")
            elif tel_env=="IS":
                # print(moderation_telemetry_request)
                
                if telemetry.tel_flag=="True":
                    # print("Inside IS telemetry")
                    # print("token_info -----  ",token_info)
                    if token_info is None:
                        token_info = {"unique_name":"None","X-Correlation-ID":"None","X-Span-ID":"None","appid":"None"}
                                           
                    # schema = {"eid": "LOG","ets": "1684415926991","ver": "3.0","mid": "LOG:89db26cc60aa4204a4e86d2808e10a58",
                    #         "actor": {
                    #         "id": token_info["unique_name"],      # "id": token_info["unique_name"]
                    #         "type": "user"},"context": {
                    #         "channel": "batch",
                    #         "pdata": {
                    #             "id": "test.rai.service",
                    #             "ver": "1.0",
                    #             "pid": "is-rai"
                    #         },"env": "test",
                    #         "cdata": [],
                    #         "rollup": {}
                    #         },"X-Correlation-ID":token_info["X-Correlation-ID"],
                    #         "X-Span-ID":token_info["X-Span-ID"],
                    #         "object": {
                    #             "id": "contentId",
                    #             "ver": 1
                    #         },"tags": [],
                    #         "edata": {
                    #             "message": timecheck,
                    #             "response" : {},
                    #             "level": "api_call",
                    #             "type": "INFO"}}
                    
# NEW SCHEMA        
                    # print("moderation_telemetry_request.dict()--->>>",moderation_telemetry_request.dict()["moderationResults"]["summary"]["status"])
                    if moderation_telemetry_request.dict()["moderationResults"]["summary"]["status"] =="PASSED":
                        response = moderation_telemetry_request.dict()["moderationResults"]["summary"]["status"]
                        timings = {"Time for each individual check": timecheck, "Time taken by each model": modeltime,"Total time for moderation Check": totaltimeforallchecks}
                        message = {"response":response,"timings":timings}

                        schema = {"eid": "LOG",
                                "ets": "1708686424895",
                                "ver": "1.0",
                                "mid": "1671857291575431168",
                                "actor": {
                                    "id": token_info["unique_name"],      # "id": token_info["unique_name"]
                                    "type": "user"},
                                "context": {
                                    "channel": "web",
                                    "pdata": {
                                            "id": os.getenv("IS_PDATA_ID"),
                                            "ver": os.getenv("IS_PDATA_VER"),
                                            "pid": os.getenv("IS_PDATA_PID")                                        
                                },
                                "env": os.getenv("IS_ENV"),
                                "did": "BLRKECXXXXXXL",
                                "cdata": [
                                    {
                                        "type": "api",
                                        "id": token_info["X-Correlation-ID"]
                                    }
                                    ]
                                },                        
                                "edata": {
                                    "message": str(message),    
                                    "params": [
                                        {                                                                        
                                            "X-Span-ID": token_info["X-Span-ID"], 
                                            "code": 200,
                                            "timetaken": totaltimeforallchecks,
                                            "method": "POST",
                                            "actor": token_info["unique_name"],
                                            "appid": token_info["appid"],
                                            "path": "/moderations"
                                        }
                                    ],                    
                                    "type": "api_call",
                                    "level": "INFO"
                                    }
                                }

                        istelemetryurl = os.getenv("IS_TELEMETRY_ENDPOINT")
                        # print("schema[edata][response]--- >>> 1 ",schema["edata"]["response"])
                        # print("moderation_telemetry_request.dict()",moderation_telemetry_request.dict())
                        
                        # schema["edata"]["response"]=moderation_telemetry_request.dict()
                        # schema["edata"]["error"]=moderation_telemetry_request.dict()
                        
                        
                        print("schema ---- >>> ",json.dumps(schema))
                        # print("schema[edata][response]--- >>> 2",schema["edata"]["response"])
                        print("Inside moderation Telemetry IS  ")
                        response=requests.post(istelemetryurl,json=schema,headers=header)
                        print("sent to telemetry from send telemetry IS")
                    
                    else:
                        response_is = {
                            "text": moderation_telemetry_request.dict()["moderationResults"]['text'],
                            "reason": moderation_telemetry_request.dict()["moderationResults"]["summary"]["reason"]
                        }
                        schema={
                            "eid": "FAILED",
                            "ets": "1708686424895",
                            "ver": "1.0",
                            "mid": "1671857291575431168",
                            "actor": {
                                "id": token_info["unique_name"],
                                "type": "user"
                                },
                            "context": {
                                "channel": "web",
                                "pdata": {
                                    "id": os.getenv("IS_PDATA_ID"),
                                    "ver": os.getenv("IS_PDATA_VER"),
                                    "pid": os.getenv("IS_PDATA_PID")
                                },
                                "env": os.getenv("IS_ENV"),
                                "did": "BLRKECXXXXXXL",
                                "cdata": [
                                    {
                                        "type": "api",
                                        "id": token_info["X-Correlation-ID"]
                                    }
                                ]
                            },
                            "edata": {
                                "err": "FAILED TO PASS MODERATION GUARDRAIL",
                                "errtype": "FAILED" ,
                                "stacktrace": str(response_is),
                                "details": { 
                                    "X-Span-ID": token_info["X-Span-ID"],                                    
                                    "code": 200,
                                    "timeTaken": totaltimeforallchecks,
                                    "method": "POST",
                                    "actor": token_info["unique_name"],
                                    "path": "/moderations"
                                }
                            }
                        }
                        # privacytelemetryurl = os.getenv("IS_TELEMETRY_ENDPOINT")
                        # schema["edata"]["err"]=moderation_telemetry_request.dict()
                        # schema["edata"]["response"]=moderation_telemetry_request
                        print("json.dumps(schema)   ---->>> ",json.dumps(schema))
                        # print('schema["edata"]["err"]---->>>> ',schema["edata"]["err"])
                        # print('schema["edata"]["error"]---->>>> ',schema["edata"]["error"])
                        # print('==',header)
                        print("Inside moderation error Telemetry IS for failing moderation layer checks")
                        istelemetryurl = os.getenv("IS_TELEMETRY_ENDPOINT")
                        response=requests.post(istelemetryurl,json=schema,headers=header)
                        
                        print("sending to telemetry from error telemetry IS ")
                        response.raise_for_status()
                        print("sent to telemetry from error telemetry  IS")

    
        
        except Exception as e:
            logobj = {"_id":id,"Telemetryerror":{"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Telemetry",
                                                   "Payload":moderation_telemetry_request,
                                                   "portfolio":portfolioName,"acount":accountName}}
            # Results.createlog(logobj)
            print(str(logobj))
            log.error("Error occured in Telemetry")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            


    def send_evalLLM_telemetry_request(evalLLM_telemetry_request,id,portfolioName=None,accountName=None):
        try:
            log.info(f"EvalLLM : Telemetry Flag inside telemetry method--> {telemetry.tel_flag}")
            # telFlagData = telemetrydb.TelemetryFlag.find({"Module":"AiCloudModeration"},{})[0]
            # tel_Falg = telFlagData["TelemetryFlag"]
            request_id_var.set(id)
            
            if tel_env!="IS":
                if telemetry.tel_flag=="True":
                    log.info("Inside Telemetry")
                    telemetry_request = evalLLM_telemetry_request
                    print("json ------ ",json.dumps(telemetry_request))
                    
                    if portfolioName:
                        telemetry_request["portfolioName"]=portfolioName
                        telemetry_request["accountName"]=accountName
                        
                    else:
                        telemetry_request["portfolioName"]="None"
                        telemetry_request["accountName"]="None"
                    
                    response = requests.post(evalLLMtelemetryurl, json=telemetry_request)
                    # print("json -- ", moderation_telemetry_request.json())
                    # print("response ----- ",response)
                    # print("json ------ ",json.dumps(moderation_telemetry_request))
                    log.info(" ------------------ sending to telemetry ----------------- ")
                    response.raise_for_status()
                    log.info(" ------------------ sent to telemetry ----------------- ")
        except Exception as e:
            logobj = {"_id":id,"Telemetryerror":{"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Telemetry for EvalLLM Checks",
                                                   "Payload":telemetry_request,
                                                   "portfolio":portfolioName,"acount":accountName}}
            log.error(str(logobj))
            log.error("Error occured in EvalLLM Telemetry")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")


    def send_telemetry_error_request(moderation_telemetry_request,id,lotNumber,portfolioName,accountName,userid,err_desc=None,header=None,token_info=None):
        try:
            log.info("send_telemetry_error_request  ---- >>>>> ")
            request_id_var.set(id)
            if tel_env == "ETA":                            
                if telemetry.tel_flag=="True":
                    index_name = f"responsible-ai-moderation_{str(datetime.date.today()).replace('/','_')}"  # this is to make sure that the index name is unique per day and it eassy to clear the data after some day
                    doc_id = moderation_telemetry_request["uniqueid"]#"1"  # if you dont want doc_id then remove this line and from the url also

                    url=os.getenv("ETA_TELEMETRY_ENDPOINT")
                    url = f"{url}/{index_name}/_doc/{doc_id}"
                    print("url--->>>",url)
                    headers = {
                        "Content-Type": "application/json",
                    }
                    username = os.getenv("ETA_TELEMETRY_USERNAME")
                    password = os.getenv("ETA_TELEMETRY_PASSWORD")
                    
                    if portfolioName:
                        moderation_telemetry_request["portfolioName"]=portfolioName
                        moderation_telemetry_request["accountName"]=accountName
                        moderation_telemetry_request["lotNumber"]=lotNumber
                        moderation_telemetry_request["userid"]=userid
                        moderation_telemetry_request["error"]=err_desc
                    else:
                        
                        moderation_telemetry_request["portfolioName"]="None"
                        moderation_telemetry_request["accountName"]="None"
                        moderation_telemetry_request["lotNumber"]="None"
                        moderation_telemetry_request["userid"]="None"
                        moderation_telemetry_request["error"]=err_desc
                    payload = {"data": moderation_telemetry_request} # this is the data that you want to send to the open search
                        #print("data --->>>",payload)
                    log.info("Inside moderation Telemetry  1 ")
                    response = requests.request(
                        "POST",
                        url,
                        headers=headers,
                        auth=HTTPBasicAuth(username, password),
                        data=json.dumps(payload),
                    )

                    if response.status_code >= 200 and response.status_code < 300:
                        print(f"Success - {response.text}")
                    else:
                        print(f"error - {response.text}")
                    log.info("--------------- Sent from moderation ETA ERROR Telemetry  2 ----------- ")
                
                
            elif tel_env=="AZURE":
                telemetryurl = os.getenv("TELEMETRY_PATH")
                if telemetry.tel_flag=="True":
                    if portfolioName:
                        # moderation_telemetry_request = moderation_telemetry_request.dict()
                        moderation_telemetry_request["portfolioName"]=portfolioName
                        moderation_telemetry_request["accountName"]=accountName
                        moderation_telemetry_request["lotNumber"]=lotNumber
                        moderation_telemetry_request["userid"]=userid
                        #print("moderation_telemetry_request--->",moderation_telemetry_request)
                        response = requests.post(telemetryurl, json=moderation_telemetry_request,verify=False)
                        log.info("Inside moderation error Telemetry  1 , and tel_env!=IS ")
                        response.raise_for_status()
                        print("sent to telemetry from error telemetry ")
                    else:
                        # moderation_telemetry_request = moderation_telemetry_request.dict()
                        response = requests.post(telemetryurl, json=moderation_telemetry_request,headers=header)
                        log.info("Inside moderation error Telemetry  2, and tel_env!=IS")
                        response.raise_for_status()
                        print("sent to telemetry from error telemetry ")
            elif tel_env=="IS":
                
                if telemetry.tel_flag=="True":
                    istelemetryurl = os.getenv("IS_TELEMETRY_ENDPOINT")
                    # moderation_telemetry_request["portfolioName"]=portfolioName
                    # moderation_telemetry_request["accountName"]=accountName
                    # moderation_telemetry_request["lotNumber"]=lotNumber
                    # moderation_telemetry_request["userid"]=userid
                    print("Inside IS ERROR telemetry")            
                    schema={
                            "eid": "ERROR",
                            "ets": "1708686424895",
                            "ver": "1.0",
                            "mid": "1671857291575431168",
                            "actor": {
                                "id": token_info["unique_name"],
                                "type": "user"
                                },
                            "context": {
                                "channel": "web",
                                "pdata": {
                                    "id": os.getenv("IS_PDATA_ID"),
                                    "ver": os.getenv("IS_PDATA_VER"),
                                    "pid": os.getenv("IS_PDATA_PID")
                                },
                                "env": os.getenv("IS_ENV"),
                                "did": "BLRKECXXXXXXL",
                                "cdata": [
                                    {
                                        "type": "api",
                                        "id": token_info["X-Correlation-ID"]
                                    }
                                ]
                            },
                            "edata": {
                                "err": err_desc[0]["Error"],
                                "errtype": err_desc[0]["Error Module"],
                                "stacktrace": str(err_desc),
                                "details": { 
                                    "X-Span-ID": token_info["X-Span-ID"],                                    
                                    "code": 500,
                                    "timeTaken": "0",
                                    "method": "POST",
                                    "actor": token_info["unique_name"],
                                    "path": "/moderations"
                                }
                            }
                        }
                    #print("json.dumps(schema)   ---->>> ",json.dumps(schema))
                    print("Inside moderation error Telemetry IS ")
                    response=requests.post(istelemetryurl,json=schema,headers=header)
                    
                    print("sending to telemetry from error telemetry IS ")
                    response.raise_for_status()
                    print("sent to telemetry from error telemetry  IS")
        
        except Exception as e:
            print("in exception block of send_telemetry_error_request")
            logobj = {"_id":id,"Telemetryerror":{"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Telemetry",
                                                   "Payload":moderation_telemetry_request,
                                                   "portfolio":portfolioName,"acount":accountName}}
            print("str(logobj)----- >>>>> ",str(logobj))
            log.error("Error occured in Telemetry")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
