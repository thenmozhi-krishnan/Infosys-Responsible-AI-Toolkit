'''
Copyright 2024 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import asyncio
from http.client import HTTPException
import json
import os
import threading
import time
import jwt
from dotenv import load_dotenv
load_dotenv()

from config.logger import CustomLogger, request_id_var
from exception.exception import completionException
from flask import request
import traceback
import uuid
from mapper.mapper import *
from service.service import Openaicompletions,Llamacompletion, coupledModeration, moderation, moderationTime
log=CustomLogger()
from flask import Blueprint 
from service.service import log_dict
from service.service import toxicity_popup, profanity_popup, privacy_popup, feedback_submit, organization_policy,show_score,EvalLlmCheck,Multimodal
from cov import Cov
from auth import Auth
from cov_llama import CovLlama
from geval import gEval
from telemetry import telemetry
import requests
from translate import Translate
from evalLLM import prompt_template


if os.getenv("DBTYPE") != "False":
    from dao.AdminDb import Results
    

app =  Blueprint('app', __name__)


# tel_env=os.getenv("TELEMETRY_ENVIRONMENT")
logcheck=os.getenv("LOGCHECK")
# telemetryurl = os.getenv("TELEMETRY_PATH") 


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__





def handle_object(obj):
    return vars(obj) 

@app.route("/health",methods=[ 'GET'])
def health():
    # print(request.args.get('log'))
    if(logcheck=="true"):
        log.info("Entered health routing method")
        log.info("Health check Success")
    return json.dumps({"status":"Health check Success"})

@app.route("/rai/v1/moderations",methods=[ 'POST'])
def generate_text():

    log.info("Entered create usecase routing method")
    st=time.time()
    try:
        log.info("before invoking create usecase service")
        id = uuid.uuid4().hex
        request_id_var.set(id)
        
        authorization = request.headers.get('authorization')
        response = None
        token_info = None
        
        payload = request.get_json()
        payload=AttributeDict(payload)
        
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"
        accountName = payload.AccountName if "AccountName" in payload else "None"
        portfolioName = payload.PortfolioName if "PortfolioName" in payload else "None"
          
        headers = {'Authorization': authorization,"id":id}
        
        if authorization != None:
            print("got auth from headers")
            decoded_token = jwt.decode(authorization.split(" ")[1], algorithms=["HS256"], options={"verify_signature": False})
            X_Correlation_ID = request.headers.get('X-Correlation-ID')
            X_Span_ID = request.headers.get('X-Span-ID')
            token_info = {"unique_name":decoded_token['unique_name'],"X-Correlation-ID":X_Correlation_ID,"X-Span-ID":X_Span_ID,"appid":decoded_token["appid"]}
            
        elif Auth.is_env_vars_present() is None:
            log.info("No valid token available.Going without token method")
        else:
            print("got token using auth url")
            tok = Auth.get_valid_bearer_token()
            headers = {'Authorization': f'Bearer {tok}',"id":id}
    
        if token_info is not None:
            response_all = moderation.completions(payload=payload,headers=headers,id=id,telemetryFlag=True,token_info=token_info)
        else:
            response_all = moderation.completions(payload=payload,headers=headers,id=id,telemetryFlag=True)     
        
        response = response_all[0]

        if response == "Prompt is Empty":
            raise completionException("400- Prompt is Empty")
        json_string = json.dumps(response, default=handle_object)
        response = json.loads(json_string)  
        er=log_dict[request_id_var.get()]
        
        # print("type ---- >>> ",type(er))
        # logobj = {"_id":id,"error":er}
        if len(er)!=0:
            # err_desc = er[0]["Error"]
            err_desc = er
            # print("error---->>> ",err_desc)
            logobj = {"_id":id,"error":er}
            thread_err = threading.Thread(target=telemetry.send_telemetry_error_request, args=(logobj,id,lotNumber,portfolioName,accountName,userid,err_desc,headers,token_info))
            thread_err.start()
            del log_dict[id]
        # print(response)
        return response
    except Exception as e:
        # print("error---->>> 1",er)
        # er=log_dict[request_id_var.get()]
        # print("error---->>> ",er)
        # logobj = {"_id":id,"error":er}
        # if len(er)!=0:
        #     thread_err = threading.Thread(target=telemetry.send_telemetry_error_request, args=(logobj,id,payload.PortfolioName,payload.AccountName))
        #     thread_err.start()
        #     del log_dict[id]
                 
        log.error(str(traceback.extract_tb(e.__traceback__)[0].lineno))
        log.info(e)


@app.route("/rai/v1/moderations/coupledmoderations",methods=[ 'POST'])
def generate_text2():
    log.info("Entered create usecase routing method")
    log.info("Couple Moderation API STARTED")
    st=time.time()
    try:
        log.info("before invoking create usecase service ")
        id = uuid.uuid4().hex
        
        request_id_var.set(id)
        authorization = request.headers.get('authorization')
        
        # headers = {'Authorization': authorization,"id":id}
        if authorization !=None:
            headers = {'Authorization': authorization,"id":id}
        else:
            tok = Auth.get_valid_bearer_token()
            if tok:
                headers = {'Authorization': f'Bearer {tok}',"id":id}
            else:
                log.info("No valid token available.")
        payload = request.get_json()
        response = coupledModeration.coupledCompletions(payload=payload,token=headers,id=id)
        json_string = json.dumps(response, default=handle_object)
        response = json.loads(json_string)
        log.info("after invoking create usecase service ")
        er=log_dict[request_id_var.get()]
        logobj = {"_id":id,"error":er}
        if len(er)!=0:
            if os.getenv("DBTYPE") != "False":
                Results.createlog(logobj)
            err_desc = er
            # print("error---->>> ",err_desc)
            logobj = {"_id":id,"error":er}
            payload=AttributeDict(payload)
            token_info = {"unique_name":"None","X-Correlation-ID":"None","X-Span-ID":"None"}
        
            thread_err = threading.Thread(target=telemetry.send_telemetry_error_request, args=(logobj,id,payload.lotNumber,payload.PortfolioName,payload.AccountName,payload.userid,err_desc,headers,token_info))
            thread_err.start()
            del log_dict[id]
        # log.debug("response : " + str(response))
        log.info("exit create usecase routing method")
        log.info(f"Total time taken=======> {time.time()-st}")

        return response
    except completionException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)
    

    
# To access admin url to retrieve Prompt Templates
@app.route("/rai/v1/moderations/getTemplates/<userId>",methods=[ 'GET'])
def get_templates(userId):
    log.info("Entered get_templates() routing method")
    try:
        log.info("before invoking create usecase service")
        id = uuid.uuid4().hex
        request_id_var.set(id)
        url = os.getenv("ADMINTEMPLATEPATH")+userId
        response = requests.get(url)

        if response.status_code == 200:
                prompt_template[userId]=response.json()['templates']
                log.info("Templates stored successfully")
        else:
           log.info("Error getting data: +",{response.status_code})
           
        log.info("after invoking create usecase service ")
        return "Templates Retrieved"
    except Exception as e:
        log.error(e.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**e.__dict__)

    

@app.route("/rai/v1/moderations/translate",methods=[ 'POST'])
def translate():
    log.info("Entered create usecase routing method")
    try:
        log.info("before invoking create usecase service ")
        st=time.time()
        id = uuid.uuid4().hex
        request_id_var.set(id)
        payload = request.get_json()
        payload=AttributeDict(payload)

        if payload.choice == "google":
            print("Inside Google Translate")
            text,language = Translate.translate(payload.Prompt)
        elif payload.choice == "azure":
            print("Inside Azure Translate")
            text,language = Translate.azure_translate(payload.Prompt)
        responseObj={"text":text,"language" :language,"timetaken":round(time.time()-st,3)}
        log.info("after invoking create usecase service")
        log.info("exit create usecase routing method")
        return responseObj
    except completionException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)


@app.route("/rai/v1/moderations/evalLLM",methods=[ 'POST'])
def uptrainLLMEvaluation():
    log.info("Entered evalllm check routing method")
    log.info("Eval LLM API STARTED")
    st=time.time()
    try:
        log.info("before invoking create usecase service ")
        id = uuid.uuid4().hex
        request_id_var.set(id)
        authorization = request.headers.get('authorization')
        headers = {'Authorization': authorization,"id":id}
        req = AttributeDict(request.get_json())
        ev = EvalLlmCheck()
        response = ev.evaluate(id,req)
        log.info("after invoking create usecase service ")
        er=log_dict[request_id_var.get()]
        logobj = {"_id":id,"error":er}
        
        if len(er)!=0:
            if os.getenv("DBTYPE") != "False":
                Results.createlog(logobj)
            err_desc = er
            print("error---->>> ",err_desc)
            logobj = {"_id":id,"error":er}
            payload=req
            token_info = {"unique_name":"None","X-Correlation-ID":"None","X-Span-ID":"None"}
            thread_err = threading.Thread(target=telemetry.send_telemetry_error_request, args=(logobj,id,payload.lotNumber,payload.PortfolioName,payload.AccountName,payload.userid,err_desc,headers,token_info))
            thread_err.start()
            del log_dict[id]
        log.info("exit create usecase routing method")
        log.info(f"Total time taken=======> {time.time()-st}s")
        return response 
    except Exception as e:
        log.error(e.__dict__)
        log.info("Evalllm routing method exception")
        raise HTTPException(**e.__dict__)
    


@app.route("/rai/v1/moderations/multimodal",methods=[ 'POST'])
def generate_image_explainability():
    log.info("Entered multimodal check routing method")
    log.info("MultiModal API STARTED")
    try:
        log.info("before invoking create usecase service ")
        st=time.time()
        id = uuid.uuid4().hex
        request_id_var.set(id)
        authorization = request.headers.get('authorization')
        headers = {'Authorization': authorization,"id":id}
        request_payload = {"Prompt":request.form['Prompt'],
                           "Image":request.files['Image'].stream,
                           "ModelName":request.form["model_name"],
                           "TemplateName":request.form["TemplateName"],
                           "Restrictedtopics":request.form["Restrictedtopics"],
                           "lotNumber":request.form["lotNumber"],
                           "userid":request.form["userid"],
                           "AccountName":request.form["AccountName"],
                           "PortfolioName":request.form["PortfolioName"]}
        
        m = Multimodal()
        response = m.check(request_payload)

        if request_payload['TemplateName'] == "Toxicity":
            print("score : ",response['toxicity score'])
        
        responseObj={"explanation":response['explanation'],
                     "evaluation check":request_payload['TemplateName'],
                     "threshold" :60,
                     "result":"PASSED",
                     "timetaken":str(round(time.time()-st,3))+"s",
                     "deployment_name":request_payload['ModelName']}
        
        if responseObj["evaluation check"] == "Toxicity":
            responseObj["score"] = response['toxicity score']
            for s in responseObj["score"]:
                if s["metricScore"] > 60:
                    responseObj["result"] = "FAILED"
                    break
        else:
            responseObj["score"] = int(response['score'])
            responseObj['result'] = "PASSED" if int(response['score']) < 60 else "FAILED"
        
        if responseObj["evaluation check"] in ["Toxicity","Restricted Topic"]:
            responseObj["category"] = response['category'] 
        
        
        # log.info("after invoking create usecase service ")
        # er=log_dict[request_id_var.get()]
        # logobj = {"_id":id,"error":er}

        # if len(er)!=0:
        #     if os.getenv("DBTYPE") != "False":
        #         Results.createlog(logobj)
        #     err_desc = er
        #     print("error---->>> ",err_desc)
        #     logobj = {"_id":id,"error":er}
        #     token_info = {"unique_name":"None","X-Correlation-ID":"None","X-Span-ID":"None"}
        #     thread_err = threading.Thread(target=telemetry.send_telemetry_error_request, args=(logobj,id,request_payload['lotNumber'],request_payload['PortfolioName'],request_payload['AccountName'],request_payload['userid'],err_desc,headers,token_info))
        #     thread_err.start()
        #     del log_dict[id]
        log.info("exit create usecase routing method")
        log.info(f"Total time taken=======> {time.time()-st}s")
        # return response
        return responseObj
        
     
    except Exception as e:
        log.error(e.__dict__)
        log.info("Multimodal routing method exception")
        raise HTTPException(**e.__dict__)



@app.route("/rai/v1/moderations/openai",methods=[ 'POST'])
def generate_text3():
    log.info("Entered create usecase routing method")
    try:
        log.info("before invoking create usecase service ")
        st=time.time()
        id = uuid.uuid4().hex
        request_id_var.set(id)
        payload = request.get_json()
        payload=AttributeDict(payload)
        interact=Openaicompletions()
        output_text,index,finish_reason = interact.textCompletion(payload.Prompt,float(payload.temperature),PromptTemplate="GoalPriority",deployment_name=payload.model_name)

        # output_text,index,finish_reason = interact.textCompletion(payload.Prompt,float(payload.temperature),SelfReminder=False,GoalPriority=False)
        # respoonseObj=Choice(text=output_text,index=index,finishReason = finish_reason)
        respoonseObj={"text":output_text,"index":index,"finishReason" :finish_reason,"timetaken":round(time.time()-st,3)}

        log.info("after invoking create usecase service")
        log.info("exit create usecase routing method")
        return respoonseObj
    except completionException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)
    
@app.route("/rai/v1/moderations/openaiCOT",methods=[ 'POST'])
def generate_text4():
    log.info("Entered create usecase routing method")
    try:
        st=time.time()
        log.info("before invoking create usecase service ")
        id = uuid.uuid4().hex
        request_id_var.set(id)
        payload = request.get_json()
        payload=AttributeDict(payload)
        if payload.model_name == "Llama":
            print("Inside Llama")
            interact = Llamacompletion()
        else:
            interact=Openaicompletions()
        output_text,index,finish_reason = interact.textCompletion(payload.Prompt,float(payload.temperature),PromptTemplate="GoalPriority",deployment_name=payload.model_name,Moderation_flag=False,COT=True)

        # output_text,index,finish_reason = interact.textCompletion(payload.Prompt,float(payload.temperature),SelfReminder=False,GoalPriority=False,Moderation_flag=False,COT=True)
        # responseObj=Choice(text=output_text,index=index,finishReason = finish_reason)
        respoonseObj={"text":output_text,"index":index,"finishReason" :finish_reason,"timetaken":round(time.time()-st,3)}
        log.info("after invoking create usecase service ")
        log.info("exit create usecase routing method")
        return respoonseObj
    except completionException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)


@app.route("/rai/v1/moderations/healthcareopenaiCOT",methods=[ 'POST'])
def generate_text4_healthcare():
    log.info("Entered create usecase routing method")
    try:
        st=time.time()
        log.info("before invoking create usecase service ")
        id = uuid.uuid4().hex
        request_id_var.set(id)
        payload = request.get_json()
        payload=AttributeDict(payload)
        payload.Prompt = payload.Prompt.replace('Provide justification for your answer.', 'Please consider the given question and reference then let us know in simple terms how the answer is derived.')
        payload.Prompt = payload.Prompt + ' ' + payload.PromptResponse
        if payload.model_name == "Llama":
            print("Inside Llama")
            interact = Llamacompletion()
        else:
            interact=Openaicompletions()
        output_text,index,finish_reason = interact.textCompletion(payload.Prompt,float(payload.temperature),PromptTemplate="GoalPriority",deployment_name=payload.model_name,Moderation_flag=False,COT=True)

        # output_text,index,finish_reason = interact.textCompletion(payload.Prompt,float(payload.temperature),SelfReminder=False,GoalPriority=False,Moderation_flag=False,COT=True)
        # responseObj=Choice(text=output_text,index=index,finishReason = finish_reason)
        respoonseObj={"text":output_text,"index":index,"finishReason" :finish_reason,"timetaken":round(time.time()-st,3)}
        log.info("after invoking create usecase service ")
        log.info("exit create usecase routing method")
        return respoonseObj
    except completionException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)



@app.route("/rai/v1/moderations/openaiTHOT",methods=[ 'POST'])
def generate_Thot():
    log.info("Entered create usecase routing method")
    try:
        st=time.time()
        log.info("before invoking create usecase service ")
        id = uuid.uuid4().hex
        request_id_var.set(id)
        payload = request.get_json()
        payload=AttributeDict(payload)
        if payload.model_name == "Llama":
            print("Inside Llama")
            interact = Llamacompletion()
        else:
            interact=Openaicompletions()
        output_text,index,finish_reason = interact.textCompletion(payload.Prompt,float(payload.temperature),PromptTemplate="GoalPriority",deployment_name=payload.model_name,Moderation_flag=False,THOT=True)

        # output_text,index,finish_reason = interact.textCompletion(payload.Prompt,float(payload.temperature),SelfReminder=False,GoalPriority=False,Moderation_flag=False,COT=True)
        # responseObj=Choice(text=output_text,index=index,finishReason = finish_reason)
        respoonseObj={"text":output_text,"index":index,"finishReason" :finish_reason,"timetaken":round(time.time()-st,3)}
        log.info("after invoking create usecase service ")
        log.info("exit create usecase routing method")
        return respoonseObj
    except completionException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)


# @router.get("/ModerationTime")
# def generate_text(authenticated: bool = Depends(authenticate_token)):
#     log.info("Entered create usecase routing method")
#     try:
#         log.info("before invoking create usecase service")
#         obj=moderationTime()
#         log.info("after invoking create usecase service ")
#         #log.debug("response : " + str(response))
#         log.info("exit create usecase routing method")
#         return obj
#     except completionException as cie:
#         log.error(cie.__dict__)
#         log.info("exit create usecase routing method")
#         raise HTTPException(**cie.__dict__)
    
@app.route("/rai/v1/moderations/ModerationTime",methods=[ 'GET'])
def generate_text5():
    log.info("Entered create usecase routing method")
    try:
        log.info("before invoking create usecase service")
        id = uuid.uuid4().hex
        request_id_var.set(id)
        obj=moderationTime()
        log.info("after invoking create usecase service ")
        #log.debug("response : " + str(response))
        log.info("exit create usecase routing method")
        return obj
    except completionException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)

@app.route("/rai/v1/moderations/setTelemetry",methods=[ 'POST'])
def settelemetry():
    log.info("Entered create usecase routing method ")   
    st=time.time()
    try:
        log.info("before invoking create usecase service ")
        id = uuid.uuid4().hex
        request_id_var.set(id)
        payload = request.data
        # print("checkpoint in telemetary func 1 -> ",payload)
        payload=AttributeDict(payload)
        telemetry.tel_flag=payload
        response = "Success"
        log.info("after invoking create usecase service ")
        # log.debug("response : " + str(response))
        log.info("exit create usecase routing method")
        log.info(f"Total time taken=======> {time.time()-st}")
        return response
    except completionException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)
    
@app.route("/rai/v1/moderations/ToxicityPopup",methods=[ 'POST'])
async def generate_text6():
    log.info("Entered create usecase routing method")
    try:
        log.info("before invoking create usecase service ")
        id = uuid.uuid4().hex
        request_id_var.set(id)
        authorization = request.headers.get('authorization')
        payload = request.get_json()
        payload=AttributeDict(payload)
        obj= await toxicity_popup(payload,authorization)
        log.info("after invoking create usecase service ")
        #log.debug("response : " + str(response))
        log.info("exit create usecase routing method")
        return obj
    except completionException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)

@app.route("/rai/v1/moderations/ProfanityPopup",methods=[ 'POST'])
def generate_text7():
    log.info("Entered create usecase routing method")
    authorization = request.headers.get('authorization')
    payload = request.get_json()
    payload=AttributeDict(payload)
    try:
        log.info("before invoking create usecase service ")
        id = uuid.uuid4().hex
        request_id_var.set(id)
        obj=profanity_popup(payload.text,authorization)
        log.info("after invoking create usecase service ")
        #log.debug("response : " + str(response))
        log.info("exit create usecase routing method")
        return obj
    except completionException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)
    
@app.route("/rai/v1/moderations/PrivacyPopup",methods=[ 'POST'])
def generate_text8():
    log.info("Entered create usecase routing method")
    payload = request.get_json()
    payload=AttributeDict(payload)
    try:
        log.info("before invoking create usecase service ")
        id = uuid.uuid4().hex
        request_id_var.set(id)
        obj=privacy_popup(payload)
        log.info("after invoking create usecase service ")
        #log.debug("response : " + str(response))
        log.info("exit create usecase routing method")
        json_string = json.dumps(obj, default=handle_object)
        obj = json.loads(json_string)
        return obj
    except completionException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)


    
@app.route("/rai/v1/moderations/feedback",methods=[ 'POST'])
def feedback():
    log.info("Entered create usecase routing method")
    payload = request.get_json()
    payload=AttributeDict(payload)
    try:
        log.info("before invoking create usecase service ")
        id = uuid.uuid4().hex
        request_id_var.set(id)
        response = feedback_submit(payload)
        log.info("after invoking create usecase service ")
        log.debug("response : " + str(response))
        log.info("exit create usecase routing method")
        return response
    except completionException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)
    
@app.route("/rai/v1/moderations/COV",methods=[ 'POST'])
def generate_text9():
    log.info("Entered create usecase routing method for cov")
    st=time.time()
    payload = request.get_json()
    payload=AttributeDict(payload)
    try:
        log.info("before invoking create usecase service ")
        id = uuid.uuid4().hex
        request_id_var.set(id)
        if payload.model_name == "Llama":
            print("Inside Llama COV")
            response = CovLlama.cov(payload.text, payload.complexity)
            log.info("after invoking create usecase service ")
            log.info("exit create usecase routing method")
            log.info(f"Total time taken=======> {time.time()-st}")
            final_respose =json.dumps(response)
            return final_respose
        else:
            response = Cov.cov(payload.text, payload.complexity, payload.model_name)
            log.info("after invoking create usecase service ")
            log.info("exit create usecase routing method")
            log.info(f"Total time taken=======> {time.time()-st}")
            promptTriggering = "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766"
            if (promptTriggering not in response) and (response != "Rate Limit Error"):
                response['original_question'] = response['original_question'].pop()
            final_respose =json.dumps(response)
            return final_respose
    except completionException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)

@app.route("/rai/v1/moderations/OrgPolicy",methods=[ 'POST'])
def org_policy():
    log.info("Entered create usecase routing method")
    st=time.time()
    authorization = request.headers.get('authorization')
    payload = request.get_json()
    payload=AttributeDict(payload)
    try:
        log.info("before invoking create usecase service ")
        id = uuid.uuid4().hex
        request_id_var.set(id)
        response = organization_policy(payload,authorization)
        log.info("after invoking create usecase service ")
        # log.debug("response : " + str(response))
        log.info("exit create usecase routing method")
        log.info(f"Total time taken=======> {time.time()-st}")
        return response
    except completionException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)
    
@app.route("/rai/v1/moderations/gEval",methods=[ 'POST'])
def Faithfullness():
    log.info("Entered create usecase routing method")
    st=time.time()
    authorization = request.headers.get('authorization')
    payload = request.get_json()
    payload=AttributeDict(payload)
    try:
        log.info("before invoking create usecase service ")
        id = uuid.uuid4().hex
        request_id_var.set(id)
        response = gEval(payload,authorization)
        log.info("after invoking create usecase service ")
        # log.debug("response : " + str(response))
        log.info("exit create usecase routing method")
        log.info(f"Total time taken=======> {time.time()-st}")
        return response
    except completionException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)

@app.route("/rai/v1/moderations/Hallucination_Check",methods=[ 'POST'])
def hallucination_check():
    log.info("Entered create usecase routing method")
    st=time.time()

    authorization = request.headers.get('authorization')
    payload = request.get_json()
    payload=AttributeDict(payload)
    try:
        log.info("before invoking create usecase service ")        
        request_id_var.set("similarity")   # id -> similaritycheck
        output_score = show_score(payload.prompt, payload.response, payload.sourcearr, authorization)
        log.info("after invoking create usecase service ")
        log.info("exit create usecase routing method")
        log.info(f"Total time taken=======> {time.time()-st}")
        return output_score
    except completionException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)
    
