'''
Copyright 2024 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import os
import re
import time
from config.logger import CustomLogger, request_id_var
# from dao.AdminDb import ProfaneWords

from datetime import datetime
import json
import requests
# import logging as log
import asyncio
import threading
import openai
import numpy as np
import nltk
import traceback
import urllib3
from mapper.mapper import *
from dotenv import load_dotenv
import textstat
from better_profanity import profanity
import aiohttp
import ssl
from smoothLLm import SMOOTHLLM
from telemetry import telemetry
from bergeron import  Bergeron
if os.getenv("DBTYPE") != "False":
    from dao.AdminDb import Results
from translate import Translate
from evalLLM import *
from openai import AzureOpenAI
import demoji
import string 
import regex
import grapheme
import base64
from io import BytesIO
from PIL import Image
from config.prompt_templates import *
from evalLLM import prompt_template
from privacy.privacy import Privacy as ps


log=CustomLogger()

def handle_object(obj):
    return vars(obj) 

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
nltk.data.path.append("data/nltk_data")

urllib3.disable_warnings()

try:

    device = "cuda"
    
    load_dotenv()
    jailbreakurl=os.getenv("JAILBREAKMODEL")
    promptInjectionurl=os.getenv("PROMPTINJECTIONMODEL")
    detoxifyurl=os.getenv("DETOXIFYMODEL")
    mpnetsimilarityurl=os.getenv("SIMILARITYMODEL")
    topicurl=os.getenv("RESTRICTEDMODEL")
    
    tel_env=os.getenv("TELEMETRY_ENVIRONMENT")
    telemetryurl = os.getenv("TELEMETRY_PATH") 
    coupledtelemetryurl=os.getenv("COUPLEDTELEMETRYPATH")
    evalLLMtelemetryurl=os.getenv("EVALLLMTELEMETRYPATH")
    startupFlag=True
    
    request_id_var.set("Startup")
   
    
    with open("data/jailbreak_embeddings.json", "r") as file:
        json_data = file.read()
        jailbreak_embeddings = json.loads(json_data)
    with open("data/refusal_embeddings.json", "r") as file:
        json_data = file.read()
        refusal_embeddings = json.loads(json_data)
    with open("data/topic_embeddings.json", "r") as file:
        json_data = file.read()
        topic_embeddings = json.loads(json_data)
    with open("data/orgpolicy_embeddings.json", "r") as file:
        json_data = file.read()
        orgpolicy_embeddings = json.loads(json_data)
    #load the json file for imappropriate emojis defined
    with open("data/inappropriate_emoji.json",  encoding="utf-8",mode="r") as emoji_file:
        data=emoji_file.read()
        emoji_data=json.loads(data)
    
    
    try:
        pass
        
    except Exception as e:
        log.info(str(e))
        log.error(str(traceback.extract_tb(e.__traceback__)[0].lineno))
    
    
    global log_dict
    log_dict={}
    

except Exception as e:
    log.error(str(traceback.extract_tb(e.__traceback__)[0].lineno))
    log.info(f"Exception: {e}")
    


request_id_var.set("Startup")


log_dict={}
async def post_request(url, data=None, json=None, headers=None, verify=False):
  """
  Performs a POST request using aiohttp.

  Args:
      url (str): The URL of the endpoint to send the request to.
      data (dict, optional): A dictionary of data to send as form-encoded data. Defaults to None.
      json (dict, optional): A dictionary of data to be JSON-encoded and sent in the request body. Defaults to None.
      headers (dict, optional): A dictionary of headers to include in the request. Defaults to None.

  Returns:
      aiohttp.ClientResponse: The response object from the server.
  """
  if(headers["Authorization"]==None):
      headers["Authorization"]="None"

  ssl_context = ssl.create_default_context()
  ssl_context.check_hostname = False
  ssl_context.verify_mode = ssl.CERT_NONE
  
  async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
    async with session.post(url, data=data, json=json, headers=headers) as response:
      
      response.raise_for_status() # Raise an exception for non-2xx status codes
      
      return await response.read()

dict_timecheck={"requestModeration": 
                    {"promptInjectionCheck": "0s", 
                    "jailbreakCheck": "0s", 
                    "toxicityCheck": "0s", 
                    "privacyCheck": "0s", 
                    "profanityCheck": "0s", "refusalCheck": "0s",
                    "restrictedtopic": "0s","textqualityCheck": "0s",
                    "customthemeCheck": "0s"}, 
                "responseModeration": 
                    {"toxicityCheck": "0s", 
                    "privacyCheck": "0s", 
                    "profanityCheck": "0s", 
                    "refusalCheck": "0s", 
                    "textrelevanceCheck": "0s", 
                    "textqualityCheck": "0s",
                    "customthemeCheck": "0s"}, "OpenAIInteractionTime": "0s"}
dictcheck={"promptInjectionCheck": "0s", 
                    "jailbreakCheck": "0s", 
                    "toxicityCheck": "0s", 
                    "privacyCheck": "0s", 
                    "profanityCheck": "0s", "refusalCheck": "0s",
                    "restrictedtopic": "0s","textqualityCheck": "0s",
                    "customthemeCheck": "0s"}
def writejson(dict_timecheck):            
    json_object = json.dumps(dict_timecheck)
    with open("data/moderationtime.json", "w") as outfile:
        outfile.write(json_object)
        
# Template based Guardrails (LLM Evaluation)
class EvalLlmCheck:
    
    def evaluateRequest(self,prompt,userId,deployment_name,template,temperature):
        data=[{'question':prompt}]
        customEval = CustomEvaluation(detection_type=template,userId=userId,deployment_name = deployment_name,temperature=temperature)         
        results = customEval.run(data)
        return results
    
    def evaluateResponse(self,prompt,userId,temperature,Prompt_Template,deployment_name,template):
        response,index,finish_reason = Openaicompletions().textCompletion(text=prompt,temperature=temperature,
                                                                              PromptTemplate=Prompt_Template,
                                                                              deployment_name=deployment_name)

        data=[{'question':prompt,'response':response}]
        try:
            customEval = CustomEvaluation(detection_type=template,userId=userId,deployment_name = deployment_name,temperature=temperature)         
            results = customEval.run(data)
            return results
        except Exception as e:
            line_number = traceback.extract_tb(e.__traceback__)[0].lineno
            log.error(f"Exception: {line_number,e}")
    

    def evaluate(self,id,payload):

        log_dict[request_id_var.get()]=[]
        prompt = payload.Prompt
        deployment_name = payload.model_name
        template = payload.template_name
        temperature = payload.temperature
        promptTemplate = payload.PromptTemplate
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"
        AccountName=payload.AccountName if "AccountName" in payload else "None"
        PortfolioName=payload.PortfolioName if "PortfolioName" in payload else "None"
        createdDate = datetime.datetime.now()

        try:
            if template.startswith("RESPONSE"):
                results = self.evaluateResponse(prompt,userid,float(temperature),promptTemplate,deployment_name,template)
            else:
                results = self.evaluateRequest(prompt,userid,deployment_name,template,float(temperature))
            
                    
            final_results = {
                            'uniqueid':id,
                            'userid': userid, 
                            'lotNumber': lotNumber, 
                            'created': str(createdDate), 
                            'model': deployment_name,
                            'moderationResults':{'response':results},
                            'evaluation_check':template,
                            'description':""
                            }
            
            if userid!="None":
                for d in prompt_template[userid]:
                    if d["templateName"]==template:
                        final_results['description']=d["description"]
            
            # if os.getenv("DBTYPE") != "False":
            #     thread=threading.Thread(target=Results.create,args=(final_results,id,str(PortfolioName),str(AccountName),userid,lotNumber))
            #     thread.start()
            #     log.info("Saved into DB")
            # log.info(f"Telemetry Flag just BEFORE TELEMETRY THREAD START--> {telemetry.tel_flag}")
            # try:
            #     log.info(f"EVALLLM TELEMETRY URL in MOD SERVICE {evalLLMtelemetryurl}")
            #     thread1 = threading.Thread(target=telemetry.send_evalLLM_telemetry_request, args=(final_results,id,str(PortfolioName), str(AccountName)))
            #     thread1.start()
            #     log.info("THREAD STARTED")
            # except Exception as e:
            #     log.error("Error starting telemetry thread: " + str(e))
            #     log.error(traceback.format_exc())
            
            return final_results
        
        except Exception as e:
            line_number = traceback.extract_tb(e.__traceback__)[0].lineno
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Eval LLM model call"})
            # log_dict[request_id_var.get()].append("Line number "+str(traceback.extract_tb(e.__traceback__)[0].lineno)+" "+str(e))
            log.error(f"Exception: {line_number,e}")



# MultiModal Functionality
class Multimodal:
    
    def encode_image(self,image):
        '''Encodes image using Base64 encoding'''
        try:
            im = Image.open(image)
            buffered = BytesIO()
            if im.format in ["JPEG","jpg","jpeg"]:
                format="JPEG"
            elif im.format in ["PNG","png"]:
                format="PNG"
            elif im.format in ["GIF","gif"]:
                format="GIF"
            elif im.format in ["BMP","bmp"]:
                format="BMP"
            im.save(buffered,format=format)
            buffered.seek(0) 
            encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return encoded_image
        except IOError:
            line_number = traceback.extract_tb(e.__traceback__)[0].lineno
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed in Multimodal check"})
            log.error(f"Error opening image file: {line_number,e}")

    
    def config(self,messages,modelName):
        if modelName == "gpt4O":
            AZURE_API_KEY = os.getenv('OPENAI_API_KEY_GPT4_O')
            AZURE_API_BASE =  os.getenv('OPENAI_API_BASE_GPT4_O')            
            AZURE_API_VERSION = os.getenv('OPENAI_API_VERSION_GPT4_O')
            deployment_name = os.getenv("OPENAI_MODEL_GPT4_O")

        client = AzureOpenAI(
                        azure_endpoint=AZURE_API_BASE,
                        api_key=AZURE_API_KEY,
                        api_version=AZURE_API_VERSION
                    )
        try:
            response = client.chat.completions.create(
                    model=deployment_name,
                    messages=messages,
                    max_tokens=500)
                
            return json.loads(response.choices[0].message.content)
        
        except openai.BadRequestError as IR:
            return {"explanation":str(IR),"score":100,"threshold":60}



    def check(self,payload):
        '''Implements image explainability using GPT-4o
        Args: Prompt, Image
        Return: response text'''
        try:
            base64_image=self.encode_image(payload['Image'])
            messages = [{"role": "user", "content": 
                        [
                         {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]}]
            
            
            if payload['TemplateName']=="Restricted Topic":
                args = {"prompt":payload['Prompt'],"topics":payload['Restrictedtopics']}
                messages[0]["content"].append({"type": "text", "text": restricted_topic_check.format(**args)})
            
            else:
                template = {"Prompt Injection":prompt_injection_check,
                            "Jailbreak":jail_break_check,
                            "Toxicity":toxicity_check,
                            "Profanity":profanity_check
                           }
                
                messages[0]["content"].append({"type": "text", "text": template[payload['TemplateName']].replace("prompt",payload['Prompt'])})
            
            return self.config(messages,payload['ModelName'])
        
    
        except Exception as e:
            line_number = traceback.extract_tb(e.__traceback__)[0].lineno
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed in Multimodal check"})
            log.error(f"Exception: {line_number,e}")
        


class PromptInjection:
    async def classify_text(self, text,headers):
        
        # headers=headers.split(" ")[1]
        # print(" ====>",headers)
        try:
            
            output=await post_request(url=promptInjectionurl,json={"text": text},headers=headers)
            output=json.loads(output.decode('utf-8'))
            modeltime = output[2]["time_taken"]
            if output[0]=='LEGIT':
                injectionscore = 1 - output[1]
            else:
                injectionscore = output[1]

            return round(injectionscore,3),modeltime
        except Exception as e:
            log.error("Error occured in PromptInjection")
            # arr=log_dict[request_id_var.get()]
            # print("arr1",arr)
            # print(request_id_var.get())
            # log_dict[request_id_var.get()] = arr
            line_number = traceback.extract_tb(e.__traceback__)[0].lineno
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at PromptInjection model call"})
            # log_dict[request_id_var.get()].append("Line number "+str(traceback.extract_tb(e.__traceback__)[0].lineno)+" "+str(e))
            log.error(f"Exception: {line_number,e}")

def text_quality(text):
    ease_score = textstat.flesch_reading_ease(text)
    grade_score = textstat.text_standard(text)
    return ease_score,grade_score

class promptResponse:
    async def promptResponseSimilarity (self,prompt,output_text,headers):
        try:
            output =await post_request(url = mpnetsimilarityurl,json={"text1": prompt,"text2": output_text},headers=headers)
            similarity=json.loads(output.decode('utf-8'))[0][0][0]
            
            # output = requests.post(url = mpnetsimilarityurl,json={"text1": prompt,"text2": output_text},headers=headers,verify=False)
            # similarity=output.json()[0][0]
            # log.info(f"Max similarity : {max(similarity)}")
            return similarity
        except Exception as e:
            log.error("Error occured in promptResponse")
         
            # line_number = traceback.extract_tb(e.__traceback__)[0].lineno
          
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at PromptInjection model call"})
      
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

class Jailbreak:
    async def identify_jailbreak(self, text,headers):

        try:
            # t=time.time()
            text_embedding =await post_request(url = jailbreakurl,json={"text": [text]},headers=headers)
            modelcalltime = json.loads(text_embedding.decode('utf-8'))[1]['time_taken']
            text_embedding=json.loads(text_embedding.decode('utf-8'))[0][0]
            
            
            similarities = []
            # st=time.time()
            for embedding in jailbreak_embeddings:
                # similarity = requests.post(url = mpnetsimilarityurl,json={"emb1":text_embedding,"emb2":embedding},verify=False).json()[0][0]
                # similarity = util.pytorch_cos_sim(text_embedding, embedding)
                dot_product = np.dot(text_embedding, embedding)
                norm_product = np.linalg.norm(text_embedding) * np.linalg.norm(embedding)
                similarity = round(dot_product / norm_product,4)
                similarities.append(similarity)
       
            return max(similarities),modelcalltime
        except Exception as e:
        
            log.error("Error occured in Jailbreak")
      
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at PromptInjection model call"})
            # log_dict[request_id_var.get()].append("Line number "+str(traceback.extract_tb(e.__traceback__)[0].lineno)+" "+str(e))
       
            log.error(f"Exception: {e}")

class Customtheme:
    async def identify_jailbreak(self, text,headers,theme=None):
        try:
            # text_embedding =await post_request(url = jailbreakurl,json={"text": [text]},headers=headers)
            # text_embedding=json.loads(text_embedding.decode('utf-8'))[0][0]
            
            # text_embedding = requests.post(url = jailbreakurl,json={"text": [text]},headers=headers,verify=False).json()[0]
            # customTheme_embeddings = [requests.post(url = jailbreakurl,json={"text": s},headers=headers,verify=False).json() for s in theme]
            theme.append(text)
            customTheme_embeddings =await post_request(url = jailbreakurl,json={"text": theme},headers=headers)
            # print("theme --->>> ",theme)
            # print("customTheme_embeddings",customTheme_embeddings)
            customTheme_embeddings_decoded = json.loads(customTheme_embeddings.decode('utf-8'))
            modelcalltime = customTheme_embeddings_decoded[1]['time_taken']
            text_embedding=customTheme_embeddings_decoded[0][-1]    
            customTheme_embeddings=customTheme_embeddings_decoded[0][:-1]
            # print("customTheme_embeddings",len(customTheme_embeddings))
            
            
            similarities = []
            for embedding in customTheme_embeddings:
                # similarity = requests.post(url = mpnetsimilarityurl,json={"emb1":text_embedding,"emb2":embedding},verify=False).json()[0][0]
                # similarity = util.pytorch_cos_sim(text_embedding, embedding)
                dot_product = np.dot(text_embedding, embedding)
                norm_product = np.linalg.norm(text_embedding) * np.linalg.norm(embedding)
                similarity = round(dot_product / norm_product,4)
                similarities.append(similarity)
            return max(similarities),modelcalltime
        except Exception as e:
            log.error("Error occured in Customtheme")
         
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Customtheme"})
       
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
    

class CustomthemeRestricted:
    def identify_jailbreak(self, text,headers,theme=None):
        try:
            text_embedding = requests.post(url = jailbreakurl,json={"text": [text]},headers=headers,verify=False).json()[0][0]
            # customTheme_embeddings = [jailbreakModel.encode(s, convert_to_tensor=True) for s in theme]
            if theme:
                embed_array = orgpolicy_embeddings
            else:
                embed_array = topic_embeddings
            similarities = []
            for embedding in embed_array:
                dot_product = np.dot(text_embedding, embedding)
                norm_product = np.linalg.norm(text_embedding) * np.linalg.norm(embedding)
                similarity = round(dot_product / norm_product,4)
                # similarity = util.pytorch_cos_sim(text_embedding, embedding)
                similarities.append(similarity)
        
            # print("1111",max(similarities).tolist()[0][0])
            return max(similarities)
        except Exception as e:
            log.error("Error occured in CustomthemeRestricted")
         
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at PromptInjection model call"})
          
            log.error(f"Exception: {e}")

class Refusal:
    async def refusal_check(self,text,headers):
        try:
            text_embedding =await post_request(url = jailbreakurl,json={"text": [text]},headers=headers)
            text_embedding=json.loads(text_embedding.decode('utf-8'))[0][0]
         
            similarities = []
            for embedding in refusal_embeddings:
                dot_product = np.dot(text_embedding, embedding)
                norm_product = np.linalg.norm(text_embedding) * np.linalg.norm(embedding)
                similarity = round(dot_product / norm_product,4)
                # similarity = util.pytorch_cos_sim(text_embedding, embedding)
                # similarity = requests.post(url = mpnetsimilarityurl,json={"emb1":text_embedding,"emb2":embedding},verify=False).json()[0][0]
                similarities.append(similarity)
            return max(similarities)
        except Exception as e:
            log.error("Error occured in Refusal")
          
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at PromptInjection model call"})
          
            # log.error(f"Exception: {e}")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

class Restrict_topic:
    async def restrict_topic(self,text,config_details,headers): 
        try:
            labels= config_details["ModerationCheckThresholds"]["RestrictedtopicDetails"]["Restrictedtopics"]
            output =await post_request(url = topicurl,json={"text": text,"labels":labels},headers=headers)
            output=json.loads(output.decode('utf-8'))
            modelcalltime = output['time_taken']
            # print("modelcalltime",modelcalltime)
            # output = requests.post(url = topicurl,json={"text": text,"labels":labels},headers=headers,verify=False)
            # output=output.json()
            d={}
            for i in range(len(labels)):
                d[output["labels"][i]] = str(round(output["scores"][i],3))
            # themecheck = CustomthemeRestricted()
            # d["CustomTheme"]=themecheck.identify_jailbreak(text,headers)
            log.debug(f"Dictionary for labels: {d}")
            return d,modelcalltime
        except Exception as e:
            log.error("Error occured in Restrict_topic")
           
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Restrict_topic model call"})
          

class Toxicity:
    async def toxicity_check(self,text,headers):
        try:
            if identifyIDP(text):
                text=text.replace('IDP','idp')
            # tokens = TreebankWordTokenizer().tokenize
            tokens = nltk.word_tokenize(text)  
            # print("len(tokens)",len(tokens))
            if len(tokens) > 400:
                chunked_texts = []
                chunk = []
                token_count = 0

                for token in tokens:
                    if token_count + len(token) <= 400:
                        chunk.append(token)
                        token_count += len(token)
                    else:
                        chunked_texts.append(' '.join(chunk))
                        chunk = [token]
                        token_count = len(token)

                # Add the last chunk if it's not empty
                
                if chunk:
                    chunked_texts.append(' '.join(chunk))
            
                toxicity_scoreslist = []
              #  print("chunked_texts",chunked_texts)
                # print("len(chunked_texts)",len(chunked_texts))
                for chunk in chunked_texts:
                    result =await post_request(url=detoxifyurl,json={"text": chunk},headers=headers)
                    result=json.loads(result.decode('utf-8'))
                    # print(len(chunk)," checking for token grater than 400 ",result)
                    modelcalltime = result["time_taken"]
                    # print("=======tox",result)
                   
                    # print("======tox",result)
                    # result = requests.post(url=detoxifyurl,json={"text": chunk},headers=headers,verify=False).json()
                    toxicity_scoreslist.append(result)
                
                toxicity_scores = {
                'toxicity': 0,
                'severe_toxicity': 0,
                'obscene': 0,
                'threat': 0,
                'insult': 0,
                'identity_attack': 0,
                'sexual_explicit': 0
                }
                
                count = 0
                
                for item in toxicity_scoreslist:
                    toxic_score_list = item['toxicScore']
                    for metric in toxic_score_list:
                        metric_name = metric['metricName']
                        metric_score = metric['metricScore']
                        toxicity_scores[metric_name] += metric_score
                    count += 1
                
                toxicity_averages = {key: value / count for key, value in toxicity_scores.items()}
                
                l=[]
                for i in toxicity_averages:
                    l.append({'metricName': i, 'metricScore': toxicity_averages[i]})
                output = {"toxicScore":l}

            else:
                output = requests.post(url=detoxifyurl,json={"text": text},headers=headers,verify=False).json()
                modelcalltime = output["time_taken"]
                # print("modelcalltime",modelcalltime)
            toxic_score = output["toxicScore"][0]["metricScore"]

            return toxic_score,output,modelcalltime
        except Exception as e:
        
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Toxicity model call"})
         
            log.error("Error occured in Toxicity")
            # log.error(f"Exception: {e}")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
    
async def toxicity_popup(payload,token):
    try:
        log.info(f"Initialising toxicity popup")
        # st = time.time()
        toxicity = Toxicity()
        headers = {'Authorization': token}
        payload = AttributeDict(payload)
        result = await toxicity.toxicity_check(payload.text,headers)
        toxic_dict=result
        status="PASSED"
        toxic_dict=toxic_dict[1]["toxicScore"]
        payload.ToxicityThreshold = AttributeDict(payload.ToxicityThreshold)
        if float(toxic_dict[0]["metricScore"])>payload.ToxicityThreshold.ToxicityThreshold:
            status="FAILED"
        if float(toxic_dict[1]["metricScore"])>payload.ToxicityThreshold.SevereToxicityThreshold:
            status="FAILED"
        if float(toxic_dict[2]["metricScore"])>payload.ToxicityThreshold.ObsceneThreshold:
            status="FAILED"
        if float(toxic_dict[3]["metricScore"])>payload.ToxicityThreshold.ThreatThreshold:
            status="FAILED"
        if float(toxic_dict[4]["metricScore"])>payload.ToxicityThreshold.InsultThreshold:
            status="FAILED"
        if float(toxic_dict[5]["metricScore"])>payload.ToxicityThreshold.IdentityAttackThreshold:
            status="FAILED"
        if float(toxic_dict[6]["metricScore"])>payload.ToxicityThreshold.SexualExplicitThreshold:
            status="FAILED"
        toxicity_dict={
            "toxicity":{"score":str(round(float(toxic_dict[0]["metricScore"]),3)),"threshold":payload.ToxicityThreshold.ToxicityThreshold},
            "severe_toxicity":{"score":str(round(float(toxic_dict[1]["metricScore"]),3)),"threshold":payload.ToxicityThreshold.SevereToxicityThreshold},
            "obscene":{"score":str(round(float(toxic_dict[2]["metricScore"]),3)),"threshold":payload.ToxicityThreshold.ObsceneThreshold},
            "threat":{"score":str(round(float(toxic_dict[3]["metricScore"]),3)),"threshold":payload.ToxicityThreshold.ThreatThreshold},
            "insult":{"score":str(round(float(toxic_dict[4]["metricScore"]),3)),"threshold":payload.ToxicityThreshold.InsultThreshold},
            "identity_attack":{"score":str(round(float(toxic_dict[5]["metricScore"]),3)),"threshold":payload.ToxicityThreshold.IdentityAttackThreshold},
            "sexual_explicit":{"score":str(round(float(toxic_dict[6]["metricScore"]),3)),"threshold":payload.ToxicityThreshold.SexualExplicitThreshold},
            "status":status
        }
        return {"toxicity":[toxicity_dict]}
    except Exception as e:
            log.error("Error occured in toxicity_popup")
            # log.error(f"Exception: {e}")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

def profanity_popup(text,headers):
    text_ogn=text
    emoji_dict=identifyEmoji(text)
    emoji_flag= emoji_dict['flag']  
    if emoji_flag:
        converted_text,privacy_text,current_emoji_dict=emojiToText(text=text,emoji_dict=emoji_dict)
        text=converted_text
    else:
        text=text_ogn
    try:
        max_chunk_size = 512
        list_ = text.split()
        length_list = len(list_)
        if length_list > 300:
            chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
            toxicity_scoreslist = []
        
            for chunk in chunks:
                result = requests.post(url=detoxifyurl,json={"text": chunk},headers=headers,verify=False).json()
                toxicity_scoreslist.append(result)
            
            toxicity_scores = {
            'toxicity': 0,
            'severe_toxicity': 0,
            'obscene': 0,
            'threat': 0,
            'insult': 0,
            'identity_attack': 0,
            'sexual_explicit': 0
            }
            
            count = 0
            
            for item in toxicity_scoreslist:
                toxic_score_list = item['toxicScore']
                for metric in toxic_score_list:
                    metric_name = metric['metricName']
                    metric_score = metric['metricScore']
                    toxicity_scores[metric_name] += metric_score
                count += 1
            
            toxicity_averages = {key: value / count for key, value in toxicity_scores.items()}
            
            l=[]
            for i in toxicity_averages:
                l.append({'metricName': i, 'metricScore': toxicity_averages[i]})
            output = {"toxicScore":l}

        else:
            output = requests.post(url=detoxifyurl,json={"text": text},headers=headers,verify=False).json()
        toxic_score = output["toxicScore"][0]["metricScore"]


        # output = requests.post(url=detoxifyurl,json={"text": text},verify=False).json()
        # toxic_score = output["toxicScore"][0]["metricScore"]
        List_profanity = []
        if toxic_score > 0.6:
            res = profanity.censor(text)
            if emoji_flag:
                res[1]=wordToEmoji(text=text_ogn,current_emoji_dict=current_emoji_dict,result=res[1])
                res[2]=profaneWordIndex(text=text_ogn,profane_list=res[1])
            for i in range(len(res[1])):
                List_profanity.append({"text": res[1][i],"insetIndex":res[2][i][0],"offsetIndex":res[2][i][1]})
        return {"profanity":List_profanity}
    except Exception as e:
            log.error("Error occured in profanity_popup")
            # log.error(f"Exception: {e}")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            
def privacy_popup(payload,headers=None):
    try:
        entityList= []
        entitiesconfigured = payload.PiientitiesConfiguredToDetect
        entitiesconfiguredToBlock = payload.PiientitiesConfiguredToBlock
        text=payload.text
        
        emoji_mod_opt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        emoji_flag=False

        if(emoji_mod_opt=="yes"):
            emoji_dict=identifyEmoji(text)
            emoji_flag= emoji_dict['flag']
            print("emoji_flag: ",emoji_flag)
            if emoji_flag:
                converted_text, privacy_text,current_emoji_dict=emojiToText(text=text,emoji_dict=emoji_dict)
        
        res = ps.textAnalyze({
                    "inputText": privacy_text if emoji_flag else text, #emoji check
                    "account": None,
                    "portfolio":None,
                    "exclusionList": None,
                    "fakeData": "false"
                    })

        result = "Passed"

        for i in res.PIIEntities:
            if i.type in entitiesconfiguredToBlock:
                result = "Block"
            entity_obj = PiiEntitiesforPopup(EntityType = i.type,
                                        beginOffset = i.beginOffset,
                                        endOffset = i.endOffset,
                                        score= i.score,
                                        value = i.responseText) 
            entityList.append(entity_obj)

        popup_obj = PrivacyPopup(entitiesToDetect = entitiesconfigured,
                    entitiesToBlock = entitiesconfiguredToBlock,
                    entitiesRecognized =entityList,
                    result = result)
        
        return PrivacyPopupResponse(privacyCheck = [popup_obj])
    
    except Exception as e:
        log.error("Error occured in privacy_popup")
        log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")


class Profanity:
    def __init__(self):
        
        self.profanity_method = "Better_profanity"

    async def recognise(self,text,headers):
        try:
            tokens = nltk.word_tokenize(text)    
            if len(tokens) > 400:
                chunked_texts = []
                chunk = []
                token_count = 0

                for token in tokens:
                    if token_count + len(token) <= 400:
                        chunk.append(token)
                        token_count += len(token)
                    else:
                        chunked_texts.append(' '.join(chunk))
                        chunk = [token]
                        token_count = len(token)

                # Add the last chunk if it's not empty
                if chunk:
                    chunked_texts.append(' '.join(chunk))
            
                toxicity_scoreslist = []
            
                for chunk in chunked_texts:
                    result =await post_request(url=detoxifyurl,json={"text": chunk},headers=headers)
                    result=json.loads(result.decode('utf-8'))
                    # result = requests.post(url=detoxifyurl,json={"text": chunk},headers=headers,verify=False).json()
                    toxicity_scoreslist.append(result)
                
                toxicity_scores = {
                'toxicity': 0,
                'severe_toxicity': 0,
                'obscene': 0,
                'threat': 0,
                'insult': 0,
                'identity_attack': 0,
                'sexual_explicit': 0
                }
                
                count = 0
                
                for item in toxicity_scoreslist:
                    toxic_score_list = item['toxicScore']
                    for metric in toxic_score_list:
                        metric_name = metric['metricName']
                        metric_score = metric['metricScore']
                        toxicity_scores[metric_name] += metric_score
                    count += 1
                
                toxicity_averages = {key: value / count for key, value in toxicity_scores.items()}
                
                l=[]
                for i in toxicity_averages:
                    l.append({'metricName': i, 'metricScore': toxicity_averages[i]})
                output = {"toxicScore":l}

            else:
                output = requests.post(url=detoxifyurl,json={"text": text},headers=headers,verify=False).json()
            toxic_score = output["toxicScore"][0]["metricScore"]
            # output = requests.post(url=detoxifyurl,json={"text": text},verify=False).json()
            # toxic_score = output["toxicScore"][0]["metricScore"]
            if toxic_score > 0.6:
                res = profanity.censor(text)
                return res[1]
            else:
                return []
        except Exception as e:
            log.error("Error occured in Profanity")
          
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Profanity"})
           
            # log.error(f"Exception: {e}")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            


     
class validation_input:
    def __init__(self,deployment_name,text,config_details,emoji_mod_opt,accountname,portfolio):
        self.timecheck = {}
        self.modeltime = {}
        #implement output emoji moderation
        #to check if emoji moderation is enabled
        self.emoji_flag=False
        if(emoji_mod_opt=="yes"):
            emoji_dict=identifyEmoji(text)
            self.emoji_flag= emoji_dict['flag']  
            if self.emoji_flag:
                self.converted_text, self.privacy_text,self.current_emoji_dict=emojiToText(text=text,emoji_dict=emoji_dict)
        self.text = text
        self.accountname = accountname
        self.portfolio = portfolio
        self.deployment_name = deployment_name
        self.config_details = config_details
        self.promptInjection_threshold = config_details['ModerationCheckThresholds'].get('PromptinjectionThreshold')
        self.Jailbreak_threshold=config_details['ModerationCheckThresholds'].get("JailbreakThreshold")
        self.Profanity_threshold = config_details['ModerationCheckThresholds'].get('ProfanityCountThreshold')
        self.ToxicityThreshold = (None if config_details['ModerationCheckThresholds'].get('ToxicityThresholds')==None else config_details['ModerationCheckThresholds']['ToxicityThresholds']["ToxicityThreshold"])
        self.RefusalThreshold = config_details["ModerationCheckThresholds"].get('RefusalThreshold')
        self.PIIenities_selectedToBlock = config_details['ModerationCheckThresholds'].get('PiientitiesConfiguredToBlock')
        self.Topic_threshold = (None if config_details['ModerationCheckThresholds'].get("RestrictedtopicDetails")==None else config_details['ModerationCheckThresholds']["RestrictedtopicDetails"]['RestrictedtopicThreshold'])
        self.SmoothLT=config_details['ModerationCheckThresholds'].get('SmoothLlmThreshold')   # added for smoothllm

        self.Checks_selected=config_details['ModerationChecks']
        self.dict_prompt = {}
        self.dict_jailbreak = {}
        self.dict_profanity = {}
        self.dict_privacy = {}
        self.dict_topic={}
        self.dict_customtheme={}
        self.dict_toxicity = {}
        self.dict_refusal={}
        self.dict_relevance={}
        self.dict_textQuality={}
        
        self.dict_smoothllm={}    # added for smoothllm
        self.dict_bergeron={}    # added for bergeron
        ######################################################################
        self.dict_toxicity['object'] =toxicityCheck(toxicityScore = [],
                                          toxicitythreshold = str(''),
                                          result = 'UNMODERATED')
        self.dict_profanity['object'] =profanityCheck(profaneWordsIdentified = [],
                                          profaneWordsthreshold = "0",
                                          result = 'UNMODERATED')   
        self.dict_topic['object'] = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        self.dict_refusal['object']=refusalCheck(refusalSimilarityScore =  "",
                                            RefusalThreshold = "",
                                            result = 'UNMODERATED')
        self.dict_relevance['object']=textRelevanceCheck(PromptResponseSimilarityScore = "")
        self.dict_textQuality['object']=textQuality(readabilityScore = "",
                                                        textGrade="") 
        self.dict_customtheme['object']=customThemeCheck(customSimilarityScore = str(''),
                                          themeThreshold = str(''),
                                          result = 'UNMODERATED')                                    
        self.dict_prompt['object']=promptInjectionCheck(injectionConfidenceScore = str(""),
                                          injectionThreshold = str(""),
                                          result = 'UNMODERATED')
        self.dict_jailbreak['object']=jailbreakCheck(jailbreakSimilarityScore = str(''),
                                          jailbreakThreshold = str(''),
                                          result = 'UNMODERATED') 
        self.dict_privacy['object']  = privacyCheck(entitiesRecognised = [],
                                               entitiesConfiguredToBlock = [],
                                               result = 'UNMODERATED')
        self.dict_smoothllm['object']= smoothLlmCheck(smoothLlmScore="",
                                                      smoothLlmThreshold="",
                                                      result='UNMODERATED')
        self.dict_bergeron['object']= bergeronCheck(text="",
                                                    result='UNMODERATED')          

    async def validate_smoothllm(self,headers):
        try:
            log.info(f"Initialising smoothllm validation")
            st = time.time()        
            #emoji check
            if self.emoji_flag:
                threshold, defense_output =  SMOOTHLLM.main(self.deployment_name,self.privacy_text, self.SmoothLT['input_pertubation'], self.SmoothLT['number_of_iteration'])
            else:
                threshold, defense_output =  SMOOTHLLM.main(self.deployment_name,self.text, self.SmoothLT['input_pertubation'], self.SmoothLT['number_of_iteration'])       
            

            self.dict_smoothllm['key'] = 'randomNoiseCheck'
            
            error_message = "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766"
            if type(threshold) == str:
                if error_message in threshold or threshold == "content_filter":
                    obj_smooth = smoothLlmCheck(
                                                smoothLlmScore ="0.85",
                                                smoothLlmThreshold= "0.6",
                                                result='FAILED')
        
                    self.dict_smoothllm['object'] = obj_smooth
                    self.dict_smoothllm['status'] = False
                    et = time.time()
                    rt = et - st
                    dictcheck["smoothLlmCheck"]=str(round(rt,3))+"s"
                    log.info(f"smoothllm run time: {rt}")
                    
                    return self.dict_smoothllm
                
                
            if threshold >= self.SmoothLT['SmoothLlmThreshold']:
                obj_smooth = smoothLlmCheck(
                                            smoothLlmScore = str(threshold),
                                            smoothLlmThreshold= str(self.SmoothLT['SmoothLlmThreshold']),
                                            result='FAILED')
    
                self.dict_smoothllm['object'] = obj_smooth
                self.dict_smoothllm['status'] = False
                et = time.time()
                rt = et - st
                dictcheck["smoothLlmCheck"]=str(round(rt,3))+"s"
                log.info(f"smoothllm run time: {rt}")
                
                return self.dict_smoothllm
            else:
                obj_smooth = smoothLlmCheck(smoothLlmScore = str(threshold),
                                            smoothLlmThreshold= str(self.SmoothLT['SmoothLlmThreshold']),
                                            result = 'PASSED')
                self.dict_smoothllm['object'] = obj_smooth
                self.dict_smoothllm['status'] = True
                et = time.time()
                rt = et - st
                dictcheck["smoothLlmCheck"]=str(round(rt,3))+"s"
                log.info(f"Smoothllm run time: {rt}")
             
                return self.dict_smoothllm
        except Exception as e:
            log.error("Failed at validate_smoothllm")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at validate_smoothllm"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            
    async def validate_bergeron(self,headers):
        try:
            log.info(f"Initialising Bergeron check validation")
            st = time.time()
            #emoji check
            if self.emoji_flag:
                berger, flag =  Bergeron.generate_final(self.deployment_name,self.privacy_text)
            else:
                berger, flag =  Bergeron.generate_final(self.deployment_name,self.text)  
            # print("flag",flag)
          
            
            self.dict_bergeron['key'] = 'advancedJailbreakCheck'
            
            if flag == "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766":
                obj_berger = bergeronCheck(
                                            text="UNDETERMINED",                                            
                                            result='PASSED')
    
                self.dict_bergeron['object'] = obj_berger
                self.dict_bergeron['status'] = True
                et = time.time()
                rt = et - st
                dictcheck["bergeronCheck"]=str(round(rt,3))+"s"
                log.info(f"Bergeron run time: {rt}")
                
                return self.dict_bergeron
            
            if flag == "FAILED":
                obj_berger = bergeronCheck(
                                            text="ADVERSARIAL",                                            
                                            result='FAILED')
    
                self.dict_bergeron['object'] = obj_berger
                self.dict_bergeron['status'] = False
                et = time.time()
                rt = et - st
                dictcheck["bergeronCheck"]=str(round(rt,3))+"s"
                log.info(f"Bergeron run time: {rt}")
                
                return self.dict_bergeron
            
            else:
                obj_berger = bergeronCheck(
                                            text="NON ADVERSARIAL",
                                            result = 'PASSED')
                self.dict_bergeron['object'] = obj_berger
                self.dict_bergeron['status'] = True
                et = time.time()
                rt = et - st
                dictcheck["bergeronCheck"]=str(round(rt,3))+"s"
                log.info(f"Bergeron run time: {rt}")                
             
                return self.dict_bergeron
            
            
        except Exception as e:
            log.error("Failed at validate_bergeron")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at validate_bergeron"})
            # log.error(f"Exception: {e}")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            

    
    async def validate_prompt(self,headers):
        try:
            log.info(f"Initialising PromptInjection validation")
            st = time.time()
            prompt_check = PromptInjection()
            injectionscore, modelcalltime = await prompt_check.classify_text(self.text,headers)
            self.modeltime["promptInjectionCheck"]=modelcalltime
            self.dict_prompt['key'] = 'promptInjectionCheck'
            if injectionscore >= self.promptInjection_threshold:
                obj_prompt = promptInjectionCheck(injectionConfidenceScore = str(round(injectionscore,2)),
                                            injectionThreshold = str(self.promptInjection_threshold),
                                            result = 'FAILED')
                self.dict_prompt['object'] = obj_prompt
                self.dict_prompt['status'] = False
                et = time.time()
                rt = et - st
                dictcheck["promptInjectionCheck"]=str(round(rt,3))+"s"
                log.debug(f"PromptInjection run time: {rt}")
                self.timecheck["promptInjectionCheck"]=str(round(rt,3))+"s"
                
                return self.dict_prompt
            else:
                obj_prompt = promptInjectionCheck(injectionConfidenceScore = str(injectionscore),
                                            injectionThreshold = str(self.promptInjection_threshold),
                                            result = 'PASSED')
                self.dict_prompt['object'] = obj_prompt
                self.dict_prompt['status'] = True
                et = time.time()
                rt = et - st
                dictcheck["promptInjectionCheck"]=str(round(rt,3))+"s"
                log.debug(f"PromptInjection run time: {rt}")
                self.timecheck["promptInjectionCheck"]=str(round(rt,3))+"s"

                return self.dict_prompt
        except Exception as e:
            log.error("Failed at validate_prompt")
           
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at validate_prompt"})
         
            # log.error(f"Exception: {e}")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            
            
    async def validate_jailbreak(self,headers):
        try:
            
        #    print("Entered validate_jailbreak")
            log.info(f"Initialising jailbreak validation")
            st = time.time()
            jailbreak = Jailbreak()
            result, modelcalltime = await jailbreak.identify_jailbreak(self.text, headers)
            self.modeltime['jailbreakCheck'] = modelcalltime
         #   print("res=",result)
        #    print("vali_jailbreak=",time.time())
            self.dict_jailbreak['key'] = 'jailbreakCheck'
            if result <= self.Jailbreak_threshold:
                obj_jailbreak = jailbreakCheck(jailbreakSimilarityScore = str(round(float(result),2)),
                                            jailbreakThreshold = str(self.Jailbreak_threshold),
                                            result = 'PASSED')
                self.dict_jailbreak['object'] = obj_jailbreak
                self.dict_jailbreak['status'] = True
                et = time.time()
                rt = et - st
                dictcheck["jailbreakCheck"]=str(round(rt,3))+"s"
                log.info(f"jailbreak run time: {rt}")
                self.timecheck["jailbreakCheck"]=str(round(rt,3))+"s"
                
                return self.dict_jailbreak
            
            else:
                obj_jailbreak = jailbreakCheck(jailbreakSimilarityScore =  str(round(float(result),2)),
                                            jailbreakThreshold = str(self.Jailbreak_threshold),
                                            result = 'FAILED')
                self.dict_jailbreak['object'] = obj_jailbreak
                self.dict_jailbreak['status'] = False
                et = time.time()

                rt = et - st
                dictcheck["jailbreakCheck"]=str(round(rt,3))+"s"
                log.debug(f"jailbreak run time: {rt}")
                self.timecheck["jailbreakCheck"]=str(round(rt,3))+"s"
                
                return self.dict_jailbreak
        except Exception as e:
            log.error("Failed at validate jailbreak")
          
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at validate jailbreak"})

            # log.error(f"Exception: {e}")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

    async def validate_customtheme(self,theme,headers):
        try:
            log.info(f"Initialising Customtheme validation")
            st = time.time()
            customtheme = Customtheme()
            result, modelcalltime = await customtheme.identify_jailbreak(self.text,headers,theme.ThemeTexts)
            self.modeltime["customthemeCheck"]=modelcalltime
            self.dict_customtheme['key'] = 'CustomThemeCheck'
            if result <= theme.Themethresold:
                obj_jailbreak = customThemeCheck(customSimilarityScore = str(round(float(result),2)),
                                            themeThreshold = str(theme.Themethresold),
                                            result = 'PASSED')
                self.dict_customtheme['object'] = obj_jailbreak
                self.dict_customtheme['status'] = True
                et = time.time()
                rt = et - st
                dictcheck["customthemeCheck"]=str(round(rt,3))+"s"
                log.info(f"jailbreak run time: {rt}")
                self.timecheck["customthemeCheck"]=str(round(rt,3))+"s"
                return self.dict_customtheme
            
            else:
                obj_jailbreak = customThemeCheck(customSimilarityScore =  str(round(float(result),2)),
                                            themeThreshold = str(theme.Themethresold),
                                            result = 'FAILED')
                self.dict_customtheme['object'] = obj_jailbreak
                self.dict_customtheme['status'] = False
                et = time.time()

                rt = et - st
                dictcheck["customthemeCheck"]=str(round(rt,3))+"s"
                log.debug(f"CustomTheme run time: {rt}")
                self.timecheck["customthemeCheck"]=str(round(rt,3))+"s"
                return self.dict_customtheme
        except Exception as e:
            log.error("Failed at validate customtheme")

            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at validate customtheme"})
        
            # log.error(f"Exception: {e}")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

        
    async def validate_profanity(self):
        try:
            log.info(f"Initialising profanity validation")
            st = time.time()
            profanity = Profanity()
            #check emoji
            if self.emoji_flag:
                result = await profanity.recognise(self.converted_text)
                #check and convert profane word back to emoji
                result=wordToEmoji(self.text,self.current_emoji_dict,result)
                
            else:
                result = await profanity.recognise(self.text)
            self.dict_profanity['key'] = 'profanityCheck'
            if len(result) < self.Profanity_threshold:
                obj_profanity = profanityCheck(profaneWordsIdentified = result,
                                            profaneWordsthreshold = str(self.Profanity_threshold),
                                            result = 'PASSED')
                self.dict_profanity['object'] = obj_profanity
                self.dict_profanity['status'] = True
                et = time.time()
                rt = et - st
                dictcheck["profanityCheck"]=str(round(rt,3))+"s"
                log.info(f"profanity run time: {rt}")
                self.timecheck["profanityCheck"]=str(round(rt,3))+"s"
            
                return self.dict_profanity
            
            else:
                obj_profanity = profanityCheck(profaneWordsIdentified = result,
                                            profaneWordsthreshold = str(self.Profanity_threshold),
                                            result = 'FAILED')
                self.dict_profanity['object'] = obj_profanity
                self.dict_profanity['status'] = False
                et = time.time()
                rt = et - st
                dictcheck["profanityCheck"]=str(round(rt,3))+"s"
                log.debug(f"profanity run time: {rt}")
                self.timecheck["profanityCheck"]=str(round(rt,3))+"s"

                return self.dict_profanity
        except Exception as e:
            log.error("Failed at validate profanity")
           
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at validate profanity"})
           
            # log.error(f"Exception: {e}")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

    # Integrating Privacy into Moderation
    async def validate_pii(self,headers):
        try:
            log.info(f"Initialising PII validation")
            st = time.time()

            res = ps.textAnalyze({
                    "inputText": self.privacy_text if self.emoji_flag else self.text, #emoji check
                    "account": None,
                    "portfolio":None,
                    "exclusionList": None,
                    "fakeData": "false"
                    })
            
            piiEntitiesDetected = [i.type for i in res.PIIEntities]
            
            self.dict_privacy['key'] = 'privacyCheck'

            if any(x in piiEntitiesDetected for x in self.PIIenities_selectedToBlock):
                obj_privacy = privacyCheck(entitiesRecognised = piiEntitiesDetected,
                                           entitiesConfiguredToBlock = self.PIIenities_selectedToBlock,
                                           result = 'FAILED')
                self.dict_privacy['status'] = False
            else:
                obj_privacy = privacyCheck(entitiesRecognised = piiEntitiesDetected,
                                           entitiesConfiguredToBlock = self.PIIenities_selectedToBlock,
                                           result = 'PASSED')
                self.dict_privacy['status'] = True
                
            
            self.dict_privacy['object'] = obj_privacy
            et = time.time()
            rt = et - st
            dictcheck["privacyCheck"]=str(round(rt,3))+"s"
            log.debug(f"PII run time: {rt}")
            self.timecheck["privacyCheck"]=str(round(rt,3))+"s"    
            self.modeltime['privacyCheck']=str(round(rt,3))+"s"
            
            return self.dict_privacy
        
        except Exception as e:
            log.error("Failed at validate pii")
          
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at validate pii"})
           
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")


    async def validate_restrict_topic(self,config_details,headers):
        try:
            log.info(f"Initialising Restricted Topic validation")
            st = time.time()
            topic = Restrict_topic()
            #emoji check
            if self.emoji_flag:
                result, modelcalltime=await topic.restrict_topic(self.converted_text,config_details,headers)
            else:
                result, modelcalltime=await topic.restrict_topic(self.text,config_details,headers)
            self.modeltime['restrictedtopic']=modelcalltime
            self.dict_topic['key'] = 'topicCheck'
            
            success=1
            for i in result:
                if float(result[i])>self.Topic_threshold:
                    success=0
            if success:
                self.dict_topic['status']= True
                obj_topic = restrictedtopic(topicScores=[result],topicThreshold=str(self.Topic_threshold),result = "PASSED")
            else:
                self.dict_topic['status']= False
                obj_topic = restrictedtopic(topicScores=[result],topicThreshold=str(self.Topic_threshold),result = "FAILED") 
            self.dict_topic['object'] = obj_topic
            rt = time.time()-st
            dictcheck["restrictedtopic"]=str(round(rt,3))+"s"
            log.debug(f"Restricted topic run time: {rt}")
            self.timecheck["restrictedtopic"]=str(round(rt,3))+"s"

            return self.dict_topic
        except Exception as e:
            log.error("Failed at validate restrictedtopic")
          
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at alidate restrictedtopic"})
         
            log.error(f"Exception: {e,str(traceback.extract_tb(e.__traceback__)[0].lineno)}")
            
    
    async def validate_toxicity(self,headers):
        try:
            log.info(f"Initialising toxicity validation")
            st = time.time()
            toxicity = Toxicity()
            #emoji check
            if self.emoji_flag:
                result,toxic_dict, modelcalltime =await toxicity.toxicity_check(self.converted_text,headers)
            else:
                result,toxic_dict, modelcalltime =await toxicity.toxicity_check(self.text,headers)
            
            self.dict_toxicity['key'] = 'toxicityCheck'
            self.modeltime['toxicityCheck']=modelcalltime
            list_toxic = []
            list_toxic.append(toxic_dict)
            rounded_toxic = []
            for item in list_toxic:
                toxic_score = item['toxicScore']
                rounded_score = [{'metricName': score['metricName'], 'metricScore': round(score['metricScore'], 3)} for score in toxic_score]
                rounded_item = {'toxicScore': rounded_score}
                rounded_toxic.append(rounded_item)
                
            if result < self.ToxicityThreshold:
                obj_toxicity = toxicityCheck(toxicityScore =rounded_toxic,
                                            toxicitythreshold = str(self.ToxicityThreshold),
                                            result = 'PASSED')
                self.dict_toxicity['object'] = obj_toxicity
                self.dict_toxicity['status'] = True
                et = time.time()
                rt = et - st
                dictcheck["toxicityCheck"]=str(round(rt,3))+"s"
                self.timecheck["toxicityCheck"]=str(round(rt,3))+"s"
                return self.dict_toxicity
            else:
                obj_toxicity = toxicityCheck(toxicityScore = list_toxic,
                                            toxicitythreshold = str(self.ToxicityThreshold),
                                            result = 'FAILED')
                self.dict_toxicity['object'] = obj_toxicity
                self.dict_toxicity['status'] = False
                et = time.time()
                rt = et - st
                dictcheck["toxicityCheck"]=str(round(rt,3))+"s"
                log.info(f"toxicity run time: {rt}")
                self.timecheck["toxicityCheck"]=str(round(rt,3))+"s"
                return self.dict_toxicity
        except Exception as e:
          #  print(e)
            log.error("Failed at validate toxicity")
           
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at validate toxicity"})
           
            # log.error(f"Exception: {e}")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
    

    async def validate_profanity(self,header):
        try:
            log.info(f"Initialising profanity validation")
            st = time.time()
            profanity = Profanity()
            #check emoji
            if self.emoji_flag:
                result = await profanity.recognise(self.converted_text,header)
                #check and convert profane word back to emoji
                result=wordToEmoji(self.text,self.current_emoji_dict,result)
                        
            else:
                result = await profanity.recognise(self.text,header)
            self.dict_profanity['key'] = 'profanityCheck'
            if len(result) < self.Profanity_threshold:
                obj_profanity = profanityCheck(profaneWordsIdentified = result,
                                            profaneWordsthreshold = str(self.Profanity_threshold),
                                            result = 'PASSED')
                self.dict_profanity['object'] = obj_profanity
                self.dict_profanity['status'] = True
                et = time.time()
                rt = et - st
                dictcheck["profanityCheck"]=str(round(rt,3))+"s"
                self.timecheck["profanityCheck"]=str(round(rt,3))+"s"
                log.debug(f"profanity run time: {rt}")
            
                return self.dict_profanity
            
            else:
                obj_profanity = profanityCheck(profaneWordsIdentified = result,
                                            profaneWordsthreshold = str(self.Profanity_threshold),
                                            result = 'FAILED')
                self.dict_profanity['object'] = obj_profanity
                self.dict_profanity['status'] = False
                et = time.time()
                rt = et - st
                dictcheck["profanityCheck"]=str(round(rt,3))+"s"
                log.debug(f"profanity run time: {rt}")
                self.timecheck["profanityCheck"]=str(round(rt,3))+"s"
                return self.dict_profanity
        except Exception as e:
            log.error("Failed at validate profanity")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at validate profanity"})
            # log.error(f"Exception: {e}")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

    
    async def validate_refusal(self,headers):
        try:
            log.info(f"Initialising Refusal validation")
            st = time.time()
            refusal = Refusal()
            result = await refusal.refusal_check(self.text,headers)
            self.dict_refusal['key'] = 'refusalCheck'
            if result <= self.RefusalThreshold:
                obj_refusal= refusalCheck(refusalSimilarityScore = str(round(float(result),2)),
                                            RefusalThreshold = str(self.RefusalThreshold),
                                            result = 'PASSED')
                self.dict_refusal['object'] = obj_refusal
                self.dict_refusal['status'] = True
                et = time.time()
                rt = et - st
                dictcheck["refusalCheck"]=str(round(rt,3))+"s"
                log.debug(f"refusal run time: {rt}")
                self.timecheck["refusalCheck"]=str(round(rt,3))+"s"
                return self.dict_refusal
            
            else:
                obj_refusal = refusalCheck(refusalSimilarityScore =  str(round(float(result),2)),
                                            RefusalThreshold = str(self.RefusalThreshold),
                                            result = 'FAILED')
                self.dict_refusal['object'] = obj_refusal
                self.dict_refusal['status'] = False
                et = time.time()
                rt = et - st
                dictcheck["refusalCheck"]=str(round(rt,3))+"s"
                log.debug(f"refusal run time: {rt}")
                self.timecheck["refusalCheck"]=str(round(rt,3))+"s"
                return self.dict_refusal
        except Exception as e:
            log.error("Failed at validate refusal")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at validate refusal"})
            # log.error(f"Exception: {e}")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

    async def validate_text_relevance(self,output_text,headers):
        try:
            log.info(f"Initialising Text relevance validation")
            st = time.time()
            self.dict_relevance['key']="textRelevanceCheck"
            prSimilarity = promptResponse()
            prSimilarityscore = await prSimilarity.promptResponseSimilarity(output_text,self.text,headers)
            self.dict_relevance['status']=True
            self.dict_relevance['object']=textRelevanceCheck(PromptResponseSimilarityScore = str(round(float(prSimilarityscore),2)))
            rt = time.time()-st
            dictcheck["textrelevanceCheck"]=str(round(rt,3))+"s"
            log.debug(f"Text relevance run time: {rt}")
            self.timecheck["textrelevanceCheck"]=str(round(rt,3))+"s"

            return self.dict_relevance
        except Exception as e:
            log.error("Failed at validate_text_relevance")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at validate_text_relevance"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

    async def validate_text_quality(self):
        try:
            log.info(f"Initialising Text quality validation")
            st = time.time()
            self.dict_textQuality['key']="textQualityCheck"
            readabilityScore,textGrade = text_quality(self.text)
            
            self.dict_textQuality['status']=True
            self.dict_textQuality['object']=textQuality(readabilityScore = str(readabilityScore),
                                                        textGrade=str(textGrade))
            et = time.time()
            rt = et - st
            dictcheck["textqualityCheck"]=str(round(rt,3))+"s"
            log.debug(f"Text quality run time: {rt}")
            self.timecheck["textqualityCheck"]=str(round(rt,3))+"s"
            return self.dict_textQuality
        except Exception as e:
            log.error("Failed at validate_text_quality")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at validate_text_quality"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
        


    async def main(self,theme,output_text,headers,llm_BasedChecks=[]):
        try:
            tasks=[]
            checkdict={
                'PromptInjection':"self.validate_prompt(headers)",
                'JailBreak':"self.validate_jailbreak(headers)",
                'Toxicity':"self.validate_toxicity(headers)",
                'Piidetct':"self.validate_pii(headers)",
                'Profanity':"self.validate_profanity(headers)",
                "CustomizedTheme":"self.validate_customtheme(theme,headers)",
                'RestrictTopic':"self.validate_restrict_topic(self.config_details,headers)",
                'Refusal' : "self.validate_refusal(headers)",
                'TextRelevance' : "self.validate_text_relevance(output_text,headers)",
                'TextQuality' : "self.validate_text_quality()",
                'randomNoiseCheck':'self.validate_smoothllm(headers)',
                'advancedJailbreakCheck':'self.validate_bergeron(headers)'
                }
            for i in self.Checks_selected:
                    tasks.append(eval(checkdict[i]))
            for i in llm_BasedChecks:
                    tasks.append(eval(checkdict[i]))

            results = await asyncio.gather(*tasks)
            
            list_tasks = []
            for i in results:
                list_tasks.append(i['status'])
            final_result = all(list_tasks)
            return final_result,results
        except Exception as e:
          #  print("=======err",e)
            log.error(f"Exception: {e}")
            log.error("Failed at Validate Main ------ ", str(traceback.extract_tb(e.__traceback__)[0].lineno))
            
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Validate Main"})
            # log.error(f"Exception: {e}")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

class moderation:
    def completions(payload,headers,id,deployment_name=None,output_text=None,result_flag=1,llm_BasedChecks=[],telemetryFlag=False,token_info=None,translate=None) -> dict:
        try:
            
            log_dict[request_id_var.get()]=[]
            payload1=payload
            log.info(f"Initialising completions functions")
            payload=AttributeDict(payload)
            st = time.time()
            created = datetime.datetime.now()
            if translate == "google":
                print("Inside Google Translate")
                starttime = time.time()
                text,lang = Translate.translate(payload.Prompt)
                endtime = time.time()
                rt = endtime - starttime
                dict_timecheck["translate"]=str(round(rt,3))+"s"
            elif translate == "azure":
                print("Inside Azure Translate")
                starttime = time.time()
                text,lang = Translate.azure_translate(payload.Prompt)
                endtime = time.time()
                rt = endtime - starttime
                dict_timecheck["translate"]=str(round(rt,3))+"s"
            else:
                text = payload.Prompt
            if(text==""):
                log.info("Prompt is Empty")
                log_dict[request_id_var.get()].append("Prompt is Empty")
                return "Error Occured due to empty prompt"    #,"Error Occured due to empty prompt","Error Occured due to empty prompt"

            userid=payload.userid if "userid" in payload else "None"
            portfolio = payload.PortfolioName if "PortfolioName" in payload else "None"
            accountname = payload.AccountName if "AccountName" in payload else "None"
            lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"
            global startupFlag,jailbreak_embeddings,refusal_embeddings,topic_embeddings
            config_details = payload1
            payload.ModerationCheckThresholds=AttributeDict(payload.ModerationCheckThresholds)
            theme=AttributeDict(payload.ModerationCheckThresholds.CustomTheme)
            #for emoji moderation
            emoji_mod_opt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
             
            log.debug(f"New Use case with id {id}")
            tt=time.time()
            validate_input=validation_input(deployment_name,text,config_details,emoji_mod_opt,accountname,portfolio)
            
            passed_text,dict_all=asyncio.run(validate_input.main(theme,output_text,headers,llm_BasedChecks))
            
            log.debug(f"Time for all checks ={time.time()-tt}")
    
            for i in dict_all:
                if i['key'] == 'promptInjectionCheck':
                    objprompt = i['object']
                if i['key'] == 'jailbreakCheck':
                    objjailbreak = i['object']
                if i['key'] == 'profanityCheck':
                    objprofanity = i['object']
                if i['key'] == 'privacyCheck':
                    objprivacy = i['object']
                if i['key'] == 'toxicityCheck':
                    objtoxicity = i['object']
                if i['key'] == 'topicCheck':
                    objtopic = i['object']
                if i["key"] == "CustomThemeCheck":
                    objcustomtheme = i['object']
                if i['key'] == "textQualityCheck":
                    objtextQuality = i['object']
                if i["key"] == "textRelevanceCheck":
                    objtextRelevance = i['object']
                if i["key"] == "randomNoiseCheck":
                    objsmoothllm = i['object']
                if i["key"] == "advancedJailbreakCheck":
                    objbergeron = i['object']
                

            objprompt = validate_input.dict_prompt['object']  ############
            objjailbreak=validate_input.dict_jailbreak['object']
            objprofanity=validate_input.dict_profanity['object']
            objprivacy=validate_input.dict_privacy['object']
            objtoxicity=validate_input.dict_toxicity['object']
            objtopic=validate_input.dict_topic['object']
            objcustomtheme=validate_input.dict_customtheme['object']
            objtextQuality=validate_input.dict_textQuality['object']
            objrefusal=validate_input.dict_refusal['object']
            objtextRelevance=validate_input.dict_relevance['object']
            objsmoothllm=validate_input.dict_smoothllm['object']
            objbergeron=validate_input.dict_bergeron['object']


            list_checks = []
            status = 'PASSED'
            for i in dict_all:
                if i['status']==False:
                    status = 'FAILED'
                    list_checks.append(i['key'])

            objSummary = summary(status = status,
                                reason = list_checks)
            
            log.debug(f'objSummary:{objSummary}')
            if passed_text==True:
               
                obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = objprompt,
                                                        jailbreakCheck= objjailbreak,
                                                        privacyCheck = objprivacy,
                                                        profanityCheck = objprofanity,
                                                        toxicityCheck = objtoxicity,
                                                        restrictedtopic = objtopic,
                                                        customThemeCheck = objcustomtheme,
                                                        textQuality =objtextQuality,
                                                        refusalCheck = objrefusal,
                                                        summary = objSummary)
                
                obj_ModerationResults = ModerationResults(lotNumber=lotNumber,uniqueid = id,created=str(created) ,moderationResults = obj_requestmoderation)
                
                resultsavetime=time.time()
                
                if telemetryFlag==True:
                    totaltimeforallchecks = str(round(time.time() - st,3))+"s"
                    thread = threading.Thread(target=telemetry.send_telemetry_request, args=(obj_ModerationResults,id,lotNumber, portfolio, accountname,userid,headers,token_info,validate_input.timecheck, validate_input.modeltime,totaltimeforallchecks))
                    thread.start()
                log.debug(f"Time taken in adding to telemetry {time.time()-resultsavetime}")
                et = time.time()
                rt = et - st
                log.debug(f'Run time completions if input passed :{rt}')
                if result_flag and os.getenv("DBTYPE") != "False":
                    
                    thread2=threading.Thread(target=Results.create,args=(obj_ModerationResults,id,portfolio, accountname,userid, lotNumber))
                    thread2.start()
                
                if output_text:
                    if not len(llm_BasedChecks)==0:
                            return obj_ModerationResults,objtextRelevance,objsmoothllm,objbergeron,validate_input
                    else:                       
                        return obj_ModerationResults,objtextRelevance,validate_input

                if not len(llm_BasedChecks)==0:                  
                    return obj_ModerationResults,objsmoothllm,objbergeron,validate_input
                
                log.info(f'Run time completions if input passed :{rt}')
                return obj_ModerationResults,validate_input

            else:   
                
                obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = objprompt,
                                                        jailbreakCheck= objjailbreak,
                                                        privacyCheck = objprivacy,
                                                        profanityCheck = objprofanity,
                                                        toxicityCheck = objtoxicity,
                                                        restrictedtopic = objtopic,
                                                        customThemeCheck = objcustomtheme,
                                                        textQuality = objtextQuality,
                                                        # textRelevance : textRelevanceCheck
                                                        refusalCheck = objrefusal,                                                 
                                                        summary = objSummary)
                obj_ModerationResults = ModerationResults(lotNumber=lotNumber, uniqueid = id,created=str(created),moderationResults = obj_requestmoderation)
                if telemetryFlag==True:
                    totaltimeforallchecks = str(round(time.time() - st,3))+"s"
                    thread = threading.Thread(target=telemetry.send_telemetry_request, args=(obj_ModerationResults,id,lotNumber, portfolio, accountname,userid,headers,token_info,validate_input.timecheck, validate_input.modeltime, totaltimeforallchecks))
                    thread.start()
                
                if result_flag and os.getenv("DBTYPE") != "False":
                    thread2=threading.Thread(target=Results.create,args=(obj_ModerationResults,id,portfolio, accountname,userid, lotNumber))
                    thread2.start()
                et = time.time()
                rt = et - st
                
                if output_text:
                    if not len(llm_BasedChecks)==0:
                            return obj_ModerationResults,objtextRelevance,objsmoothllm,objbergeron,validate_input
                    else:                       
                        return obj_ModerationResults,objtextRelevance,validate_input

                if not len(llm_BasedChecks)==0:                  
                    return obj_ModerationResults,objsmoothllm,objbergeron,validate_input
                
                
                log.info(f'Run time completions if input rejected :{rt}')
                return obj_ModerationResults,validate_input
        except Exception as e:
            log.error("Failed at Completion Function")
            print("[request_id_var.get()]  ---  ",request_id_var.get())
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Cmpletion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")



class Llamacompletion:
    def __init__(self):
        self.url = os.environ.get("LLAMA_ENDPOINT")

    def textCompletion(self,text,temperature=0.1,PromptTemplate="GoalPriority",deployment_name=None,Moderation_flag=None,COT=None,THOT=None):
        if temperature == 0:
            temperature = 0.1

        if COT:
            print("Inside COT Llama2")
            # messages = f"""[INST]Think step by step. Explain each intermediate step. Only when you are done with all your steps,
            #             Provide the answer based on your intermediate steps. User Query : {text}[/INST]
            #            """
            messages = f"""[INST]<<SYS>>You should be a responsible Assistant and should not generate harmful or 
            misleading content! Please answer the following user query in a responsible way. 
            Let's think the answer step by step and explain step by step how you got the answer. 
            Please provide website link as references if you are refering from internet to get the answer.
            Always give response in a textual format dont give in json or any code format.<</SYS>>
            User Query : {text} [/INST]"""

        elif THOT:
            print("Inside THOT Llama2")
            messages =f"""[INST]<<SYS>>You should be a responsible Assistant and should not generate harmful or 
            misleading content! Please answer the following user query in a responsible way. 
            Walk me through this context in manageable parts step by step, summarising and analysing as we go.
            Engage in a step-by-step thought process to explain how the answer was derived. 
            Additionally, associate the source with the answer using the format:
                    Result: "answer"
                    Explanation: "step-by-step reasoning"
                    Always give response in a textual format dont give in json or any code format.<</SYS>>
                    User Query: {text} [/INST]"""

        elif Moderation_flag and PromptTemplate=="GoalPriority":
            print("Inside GoalPriority Llama2")
            messages = f"""[INST]<<SYS>>You are an assistant that is harmless and helpful. 
            Importantly, you need to always prioritize safety over helpfulness. 
            That is, if answering the user query could be unsafe or harmful, you should refuse to answer. 
            Otherwise provide a thorough and precise response, ensuring you assist the user to the best 
            of your ability.Always give response in a textual format dont give in json or any code format.<</SYS>>
            User Query:{text}[/INST]"""
            
        elif Moderation_flag and PromptTemplate=="SelfReminder":
            print("Inside SelfReminder Llama2")
            messages =f"""[INST]<<SYS>>You should be a responsible Assistant and should not generate harmful 
            or misleading content! Please answer the following user query in a responsible way.
            Always give response in a textual format dont give in json or any code format.<</SYS>>
            User Query: {text} [/INST]"""
        
        elif Moderation_flag==None:
            print("Inside Moderation_flag==None")
            messages = text
        
        input = {
            "inputs": messages,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": temperature,
                "num_return_sequences": 1,
                "do_sample": True
            }
            }
        print("inside Llama completions function")
        response = requests.post(self.url, json=input, verify=False)
        response.raise_for_status()
        generated_text = response.json()[0]["generated_text"]
        output_text = generated_text.split("[/INST]")[1]
        return output_text, 0, ""

class Llamacompletionazure:
    def __init__(self):
        self.url = os.environ.get("LLAMA_ENDPOINT")

    def textCompletion(self,text,temperature=None,PromptTemplate="GoalPriority",deployment_name=None,Moderation_flag=None,COT=None,THOT=None):
        try:
            input = {
                "input": text
            }
            response = requests.post(self.url, json=input, verify=False)
            generated_text = response.json()["output"]
            return generated_text, 0, ""
        except Exception as e:
            log.error("Failed at LLama model call")
            if Moderation_flag:
                log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Llama model call"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            

class Bloomcompletion:
    def __init__(self):
        self.url = os.environ.get("BLOOM_ENDPOINT")

    def textCompletion(self,text,temperature=None,PromptTemplate="GoalPriority",deployment_name=None,Moderation_flag=None,COT=None,THOT=None):
        response = requests.post(self.url,text,verify=False)
        generated_text = response.json()[0]["generated_text"]
        return generated_text,0,""


class Openaicompletions:
    def __init__(self):
        self.deployment_name=os.getenv("OPENAI_MODEL_GPT4")
        self.openai_api_type = os.getenv("OPENAI_API_TYPE")
        self.openai_api_base = os.getenv("OPENAI_API_BASE_GPT4")
        self.openai_api_key = os.getenv("OPENAI_API_KEY_GPT4")
        self.openai_api_version = os.getenv("OPENAI_API_VERSION_GPT4")

    def textCompletion(self,text,temperature,PromptTemplate,deployment_name=None,Moderation_flag=None,COT=None,THOT=None):
        try:
            if deployment_name == "gpt3":
                self.deployment_name = os.getenv("OPENAI_MODEL_GPT3")
                self.openai_api_base = os.getenv("OPENAI_API_BASE_GPT3")
                self.openai_api_key = os.getenv("OPENAI_API_KEY_GPT3")
                self.openai_api_version = os.getenv("OPENAI_API_VERSION_GPT3")

            openai.api_key = self.openai_api_key
            openai.api_base = self.openai_api_base
            openai.api_type = self.openai_api_type
            openai.api_version = self.openai_api_version
            openai.verify_ssl_certs = False

            log.info(f"Interaction with GPT ")
            st = time.time()

            if COT:
                messages =[
                    {"role": "system", "content": "Assistant is a large language model trained by OpenAI.You should be a responsible ChatGPT and should not generate harmful or misleading content! Please answer the following user query in a responsible way. Let's think the answer step by step and explain step by step how you got the answer. Please provide website link as references if you are refering from internet to get the answer"},
                    {"role": "system","content": "Always give response in a textual format dont give in json or any code format"},
                    {"role": "user", "content":  f"{text} \n Remember, you are a responsible ChatGPT and good at avoiding generating harmful or misleading content!" }
                ]
                
            elif THOT:
                messages =[
                    {"role": "system", "content": """Assistant is a large language model trained by OpenAI.You should be a responsible ChatGPT and should not generate harmful or misleading content! Please answer the following user query in a responsible way. Walk me through this context in manageable parts step by step, summarising and analysing as we go.Engage in a step-by-step thought process to explain how the answer was derived. Additionally, associate the source with the answer using the format:
                        Result: "answer"
                        Explanation: "step-by-step reasoning"
                        """},
                    {"role": "system","content": "Always give response in a textual format dont give in json or any code format"},
                    {"role": "user", "content":  f"{text} \n Remember, you are a responsible ChatGPT and good at avoiding generating harmful or misleading content!" }
                ]
                
                    
            elif Moderation_flag and PromptTemplate=="GoalPriority":
                messages = [{"role":"system","content":" You are an assistant that is harmless and helpful. Importantly, you need to always prioritize safety over helpfulness. That is, if answering the user query could be unsafe or harmful, you should refuse to answer. Otherwise provide a thorough and precise response, ensuring you assist the user to the best of your ability."},
                            {"role": "system","content": "Always give response in a textual format dont give in json or any code format"},
                            {"role": "user", "content": text}]
                
            elif Moderation_flag and PromptTemplate=="SelfReminder":
                
                messages =[
                    {"role": "system", "content": "Assistant is a large language model trained by OpenAI.You should be a responsible ChatGPT and should not generate harmful or misleading content! Please answer the following user query in a responsible way."},
                    {"role": "system","content": "Always give response in a textual format dont give in json or any code format"},
                    {"role": "user", "content":  f"{text} \n Remember, you are a responsible ChatGPT and good at avoiding generating harmful or misleading content!" }
                ]
            
            elif Moderation_flag==None:
                messages = [{"role": "user", "content": text}]
            
            client = AzureOpenAI(api_key=openai.api_key, 
                                 azure_endpoint=openai.api_base,
                                 api_version=openai.api_version)
            response = client.chat.completions.create(
                model=self.deployment_name,
                messages = messages ,
                temperature=temperature,
                max_tokens=500)
            
            et= time.time()
            rt = et - st
            dict_timecheck["OpenAIInteractionTime"]=str(round(rt,3))+"s"
            log.info(f'Run time with openAI:{rt}')

            if len(response.choices[0].message.content)!=0:
                text = response.choices[0].message.content
                index = response.choices[0].index
                finish_reason= response.choices[0].finish_reason

            else:
                text = response.choices[0].finish_reason
                index = response.choices[0].index
                finish_reason = response.choices[0].finish_reason
            
            return text,index,finish_reason
        except openai.BadRequestError as IR:
            log.error(f"Exception: {IR}")
            log.error(f"Exception: {str(traceback.extract_tb(IR.__traceback__)[0].lineno),IR}")
            return str(IR),0,str(IR)
        except Exception as e:
            log.error("Failed at Openai model call")
            if Moderation_flag:
                log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Openai model call"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            return "",0,"No response from Openai"


class coupledModeration:
    def coupledCompletions(payload,token,id):
        try:
            
            created = datetime.datetime.now()
            # x=telemetry.tel_flag
            # telemetry.tel_flag=False
            global dictcheck
            # id = uuid.uuid4().hex
            # headers = {'Authorization': token}
           
            payload = AttributeDict(payload)
            st = time.time()
            
            llm_Based_Checks = payload.llm_BasedChecks

            userid = payload.userid if "userid" in payload else "None"
            lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"
            AccountName=payload.AccountName if "AccountName" in payload else "None"
            PortfolioName=payload.PortfolioName if "PortfolioName" in payload else "None"
            
            inputpayload = completionRequest(AccountName=payload.AccountName if "AccountName" in payload else "None",
                            PortfolioName=payload.PortfolioName if "PortfolioName" in payload else "None",
                            Prompt=payload.Prompt,
                            ModerationChecks=payload.InputModerationChecks,
                            ModerationCheckThresholds=payload.ModerationCheckThresholds
                            )
            
            json_string = json.dumps(inputpayload, default=handle_object)
            inputpayload = json.loads(json_string)
            emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
            inputpayload["EmojiModeration"]=emojiModOpt
            # deployment_name = payload.model_name #os.environ.get("MODEL_NAME")
            deployment_name = payload.model_name if "model_name" in payload else "gpt4"
            translate = payload.translate
            
            PromptTemplate=payload.PromptTemplate

            temperature = float(payload.temperature)
            LLMinteraction = payload.LLMinteraction
            smoothllmresponse = smoothLlmCheck(smoothLlmScore="",
                                               smoothLlmThreshold = "",
                                               result = 'UNMODERATED')
            bergerResponse = bergeronCheck(text="",
                                            result = 'UNMODERATED'
                                            )
            dict_timecheck["translate"]="0s"
            if not len(llm_Based_Checks)==0:
                
                outp1, smoothllmresponse, bergerResponse,validate_input  = moderation.completions(payload=inputpayload,headers=token,id=id,deployment_name=deployment_name,output_text=None,result_flag=0,llm_BasedChecks=llm_Based_Checks,telemetryFlag=False,translate=translate)
                request_checks = {'Time taken by each model in requestModeration' : validate_input.modeltime}
            else:
                outp1,validate_input = moderation.completions(payload=inputpayload,headers=token,id=id,deployment_name=deployment_name,output_text=None,result_flag=0,telemetryFlag=False,translate=translate)
                request_checks = {'Time taken by each model in requestModeration' : validate_input.modeltime}

            dict_timecheck["requestModeration"]= dictcheck
            dictcheck={"promptInjectionCheck": "0s", 
                        "jailbreakCheck": "0s", 
                        "toxicityCheck": "0s", 
                        "privacyCheck": "0s", 
                        "profanityCheck": "0s", "refusalCheck": "0s",
                        "restrictedtopic": "0s","textqualityCheck": "0s",
                        "customthemeCheck": "0s",
                        "smoothLlmCheck":"0s",
                        "bergeronCheck":"0s"
                        }    # added
            if outp1 == "Error Occured due to empty prompt":
                return "Empty prompt for moderation"
            elif outp1.moderationResults.summary.status =="FAILED":
                objprofanity_out = profanityCheck(profaneWordsIdentified=[],
                                                    profaneWordsthreshold = '0',
                                                    result = 'UNMODERATED')

                objprivacy_out = privacyCheck(entitiesRecognised=[],
                                            entitiesConfiguredToBlock = [],
                                            result = 'UNMODERATED')

                objtoxicity_out = toxicityCheck(toxicityScore= [],
                                                toxicitythreshold = '',
                                                result = 'UNMODERATED')

                objSummary_out = summary(status = 'Rejected',
                                        reason = ['Input Moderation Failed'])

                list_choices = []
                obj_choices = Choice(text='',
                                index= 0,
                                finishReason = '')
                list_choices.append(obj_choices)
                objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
                objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
                dict_timecheck["responseModeration"]= dictcheck
                dict_timecheck["OpenAIInteractionTime"]="0s"
                objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
                objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')

                obj_responsemoderation = ResponseModeration(generatedText = "",
                                                        privacyCheck = objprivacy_out,
                                                        profanityCheck = objprofanity_out,
                                                        toxicityCheck = objtoxicity_out,
                                                        restrictedtopic = objtopic_out,
                                                        textQuality = objtextQuality_out,
                                                        textRelevanceCheck = objpromptResponse_out,
                                                        refusalCheck = objrefusal_out,
                                                        summary = objSummary_out).__dict__
                obj_requestmoderation = CoupledRequestModeration(text = payload.Prompt,
                                                    promptInjectionCheck = outp1.moderationResults.promptInjectionCheck,
                                                    jailbreakCheck = outp1.moderationResults.jailbreakCheck,
                                                    privacyCheck = outp1.moderationResults.privacyCheck,
                                                    profanityCheck = outp1.moderationResults.profanityCheck,
                                                    toxicityCheck = outp1.moderationResults.toxicityCheck,
                                                    restrictedtopic = outp1.moderationResults.restrictedtopic,
                                                    textQuality = outp1.moderationResults.textQuality,
                                                    customThemeCheck = outp1.moderationResults.customThemeCheck,
                                                    refusalCheck = outp1.moderationResults.refusalCheck,
                                                    randomNoiseCheck = smoothllmresponse,
                                                    advancedJailbreakCheck = bergerResponse,                                                   
                                                    summary = outp1.moderationResults.summary).__dict__

                objmoderation = CoupledModerationResults(requestModeration = obj_requestmoderation,
                                            responseModeration = obj_responsemoderation)
                final_obj = completionResponse(uniqueid=id,
                                                userid=userid,
                                                lotNumber=str(lotNumber),
                                                object = "text_completion",
                                                created = str(created),
                                                model= deployment_name,
                                                choices=list_choices,
                                                moderationResults=objmoderation)
                writejson(dict_timecheck)
                # telemetry.tel_flag=x
                log.info(f"Telemetry Flag just BEFORE TELEMETRY THREAD START--> {telemetry.tel_flag}")
                try:
                    totaltimeforallchecks = str(round(time.time() - st,3))+"s"
                    log.info(f"COUPLED TELEMETRY URL in MOD SERVICE {coupledtelemetryurl}")
                    
                    dict_timecheck.update(request_checks)
                    response_checks = {"Time taken by each model in responseModeration" : 
                                       {"toxicityCheck": "0s","privacyCheck": "0s","restrictedtopic": "0s"}
                                       }
                    dict_timecheck.update(response_checks)
                    dict_timecheck.update({"Total time for moderation Check": totaltimeforallchecks})
                    thread = threading.Thread(target=telemetry.send_coupledtelemetry_request, args=(final_obj,id,str(PortfolioName), str(AccountName),dict_timecheck))
                    thread.start()
                    log.info("THREAD STARTED")
                except Exception as e:
                    log.error("Error starting telemetry thread: " + str(e))
                    log.error(traceback.format_exc())
                
                if os.getenv("DBTYPE") != "False":
                    thread2=threading.Thread(target=Results.create,args=(final_obj,id,str(PortfolioName), str(AccountName),userid,lotNumber))
                    thread2.start()
                return final_obj
            
            elif outp1.moderationResults.summary.status =="PASSED" and (LLMinteraction=="yes" or LLMinteraction=="Yes"):
                
                text = payload.Prompt
                
                if deployment_name == "Bloom":
                    print("Inside Bloom")
                    interact = Bloomcompletion()
                elif deployment_name == "Llama":
                    print("Inside Llama")
                    interact = Llamacompletion()
                elif deployment_name == "Llamaazure":
                    print("Inside Llamaazure")
                    interact = Llamacompletionazure()
                else:
                    interact=Openaicompletions()
                output_text,index,finish_reason = interact.textCompletion(text,temperature,PromptTemplate,deployment_name,1)
                list_choices = []
                obj_choices = Choice(text=output_text,
                                index= index,
                                finishReason = finish_reason)
                list_choices.append(obj_choices)
                outputpayload = completionRequest(AccountName=payload.AccountName if "AccountName" in payload else "None",
                            PortfolioName=payload.AccountName if "AccountName" in payload else "None",
                            Prompt=output_text,
                            ModerationChecks=payload.OutputModerationChecks,
                            ModerationCheckThresholds=payload.ModerationCheckThresholds)
                json_string = json.dumps(outputpayload, default=handle_object)
                outputpayload = json.loads(json_string)
                outputpayload["EmojiModeration"]=emojiModOpt
                outp2,relobj,validate_input = moderation.completions(payload=outputpayload,headers=token,id=id,deployment_name=deployment_name,output_text=text,result_flag=0,telemetryFlag=False,translate=translate)
                response_checks = {'Time taken by each model in responseModeration' : validate_input.modeltime}

                dict_timecheck["responseModeration"]= dictcheck
                obj_requestmoderation = CoupledRequestModeration(text = payload.Prompt,
                                            promptInjectionCheck = outp1.moderationResults.promptInjectionCheck,
                                            jailbreakCheck = outp1.moderationResults.jailbreakCheck,
                                            privacyCheck = outp1.moderationResults.privacyCheck,
                                            profanityCheck = outp1.moderationResults.profanityCheck,
                                            toxicityCheck = outp1.moderationResults.toxicityCheck,
                                            restrictedtopic = outp1.moderationResults.restrictedtopic,
                                            textQuality = outp1.moderationResults.textQuality,
                                            customThemeCheck = outp1.moderationResults.customThemeCheck,
                                            refusalCheck = outp1.moderationResults.refusalCheck,
                                            randomNoiseCheck = smoothllmresponse,
                                            advancedJailbreakCheck = bergerResponse,
                                            summary = outp1.moderationResults.summary).__dict__
                obj_responsemoderation = ResponseModeration(generatedText = output_text,
                                                        privacyCheck = outp2.moderationResults.privacyCheck,
                                                        profanityCheck = outp2.moderationResults.profanityCheck,
                                                        toxicityCheck = outp2.moderationResults.toxicityCheck,
                                                        restrictedtopic = outp2.moderationResults.restrictedtopic,
                                                        textQuality = outp2.moderationResults.textQuality,
                                                        textRelevanceCheck = relobj,
                                                        refusalCheck = outp2.moderationResults.refusalCheck,
                                                        summary = outp2.moderationResults.summary).__dict__
                objmoderation = CoupledModerationResults(requestModeration = obj_requestmoderation,
                                            responseModeration = obj_responsemoderation)
                final_obj = completionResponse(uniqueid=id,    
                                                object = "text_completion",
                                                userid=userid,
                                                lotNumber=str(lotNumber),
                                                created = str(created),
                                                model= deployment_name,
                                                choices=list_choices,
                                                moderationResults=objmoderation)
                
                writejson(dict_timecheck)
                # telemetry.tel_flag=x
                
                log.info(f"Telemetry Flag just BEFORE TELEMETRY THREAD START--> {telemetry.tel_flag}")
                try:
                    totaltimeforallchecks = str(round(time.time() - st,3))+"s"
                    log.info(f"COUPLED TELEMETRY URL in MOD SERVICE {coupledtelemetryurl}")
                    
                    dict_timecheck.update(request_checks)
                    if response_checks != None:
                        dict_timecheck.update(response_checks)
                    dict_timecheck.update({"Total time for moderation Check": totaltimeforallchecks})
                    thread = threading.Thread(target=telemetry.send_coupledtelemetry_request, args=(final_obj,id,outputpayload["PortfolioName"], outputpayload["AccountName"],dict_timecheck))
                    thread.start() 
                    log.info("THREAD STARTED")
                except Exception as e:
                    log.error("Error starting telemetry thread: " + str(e))
                    log.error(traceback.format_exc())
                if os.getenv("DBTYPE") != "False":
                    thread2=threading.Thread(target=Results.create,args=(final_obj,id,outputpayload["PortfolioName"], outputpayload["AccountName"],userid,lotNumber))
                    thread2.start()
                return final_obj
            else:
                objprofanity_out = profanityCheck(profaneWordsIdentified=[],
                                                profaneWordsthreshold = '0',
                                                result = 'UNMODERATED')

                objprivacy_out = privacyCheck(entitiesRecognised=[],
                                            entitiesConfiguredToBlock = [],
                                            result = 'UNMODERATED')

                objtoxicity_out = toxicityCheck(toxicityScore= [],
                                                toxicitythreshold = '',
                                                result = 'UNMODERATED')

                objSummary_out = summary(status = 'Rejected',
                                        reason = ['LLM Interaction is disabled'])

                list_choices = []
                obj_choices = Choice(text='',
                                index= 0,
                                finishReason = '')
                list_choices.append(obj_choices)
                dict_timecheck["responseModeration"]= dictcheck
                dict_timecheck["OpenAIInteractionTime"]="0s"
                objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
                objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
                objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
                objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')

                obj_responsemoderation = ResponseModeration(generatedText = "",
                                                        privacyCheck = objprivacy_out,
                                                        profanityCheck = objprofanity_out,
                                                        toxicityCheck = objtoxicity_out,
                                                        restrictedtopic = objtopic_out,
                                                        textQuality = objtextQuality_out,
                                                        textRelevanceCheck = objpromptResponse_out,
                                                        refusalCheck = objrefusal_out,
                                                        summary = objSummary_out).__dict__
                obj_requestmoderation = CoupledRequestModeration(text = payload.Prompt,
                                                    promptInjectionCheck = outp1.moderationResults.promptInjectionCheck,
                                                    jailbreakCheck = outp1.moderationResults.jailbreakCheck,
                                                    privacyCheck = outp1.moderationResults.privacyCheck,
                                                    profanityCheck = outp1.moderationResults.profanityCheck,
                                                    toxicityCheck = outp1.moderationResults.toxicityCheck,
                                                    restrictedtopic = outp1.moderationResults.restrictedtopic,
                                                    textQuality = outp1.moderationResults.textQuality,
                                                    customThemeCheck = outp1.moderationResults.customThemeCheck,
                                                    refusalCheck = outp1.moderationResults.refusalCheck,
                                                    randomNoiseCheck = smoothllmresponse,
                                                    advancedJailbreakCheck = bergerResponse,
                                                    summary = outp1.moderationResults.summary).__dict__

                objmoderation = CoupledModerationResults(requestModeration = obj_requestmoderation,
                                            responseModeration = obj_responsemoderation)
                final_obj = completionResponse(uniqueid=id,
                                                userid=userid,
                                                lotNumber=str(lotNumber),
                                                object = "text_completion",
                                                created = str(created),
                                                model= deployment_name,
                                                choices=list_choices,
                                                moderationResults=objmoderation)

                if os.getenv("DBTYPE") != "False":
                    thread2=threading.Thread(target=Results.create,args=(final_obj,id,str(PortfolioName),str(AccountName),userid,lotNumber))
                    thread2.start()
                # telemetry.tel_flag=x
                
                log.info(f"Telemetry Flag just BEFORE TELEMETRY THREAD START--> {telemetry.tel_flag}")
                try:
                    totaltimeforallchecks = str(round(time.time() - st,3))+"s"
                    log.info(f"COUPLED TELEMETRY URL in MOD SERVICE {coupledtelemetryurl}")
                    
                    dict_timecheck.update(request_checks)
                    if response_checks != None:
                        dict_timecheck.update(response_checks)
                    dict_timecheck.update({"Total time for moderation Check": totaltimeforallchecks})
                    thread = threading.Thread(target=telemetry.send_coupledtelemetry_request, args=(final_obj,id,str(PortfolioName), str(AccountName),dict_timecheck))
                    thread.start()
                    log.info("THREAD STARTED")
                except Exception as e:
                    log.error("Error starting telemetry thread: " + str(e))
                    log.error(traceback.format_exc())
                writejson(dict_timecheck)
                return final_obj
        except Exception as e:
            log.error("Failed at Coupled Completion Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Coupled Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
        
def moderationTime():
    try:
        with open("data/moderationtime.json", "r") as openfile:
            json_object = json.load(openfile)
        # print("json_object:",json_object)
        return json_object
    except Exception as e:
        print(e)
        print("Moderation time check Failed")

def feedback_submit(feedback):
    user_id = feedback.user_id
    message = feedback.message
    rating = feedback.rating
    
    res = Results.findOne(user_id)
    res["message"] = message
    res["rating"] = rating
    Results.delete(user_id)
    Results.createwithfeedback(res)
    # print("Result from db",type(Results.findOne(user_id)))
    # Process the feedback as needed

    return "Feedback submitted successfully"

def organization_policy(payload,headers): 
    try:
        labels = payload.labels
        text = payload.text
        output = requests.post(url = topicurl,json={"text": text,"labels":labels},headers=headers,verify=False)
        output=output.json()
        d={}
        for i in range(len(labels)):
            d[output["labels"][i]] = str(round(output["scores"][i],3))
        themecheck = CustomthemeRestricted()
        print("d",d)
        d["CustomTheme"]=str(themecheck.identify_jailbreak(text,headers,orgpolicy_embeddings))
        log.info(f"Dictionary for labels: {d}")
        return d
    except Exception as e:
        log.error("Error occured in Restrict_topic")
        # log.error(f"Exception: {e}")
        log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

def promptResponseSimilarity(text_1, text_2,headers):
    text_1_embedding = requests.post(url = jailbreakurl,json={"text": [text_1]},headers=headers,verify=False).json()[0][0]
    text_2_embedding = requests.post(url = jailbreakurl,json={"text": [text_2]},headers=headers,verify=False).json()[0][0]
    
    dot_product = np.dot(text_1_embedding, text_2_embedding)
    norm_product = np.linalg.norm(text_1_embedding) * np.linalg.norm(text_2_embedding)
    similarity = round(dot_product / norm_product,4)
            
    return similarity
    

def show_score(prompt, response, sourcearr,headers):
    try:        
        log.info("Showing Scores")        
        

        response = response.strip('.')  
        response=",".join(response.split(",")[:-1])
        responseArr = re.split(r'(?<=[.!?])\s+(?=\D|$)', response)

        inpoutsim = promptResponseSimilarity(prompt, response, headers)
  
        maxScore = 0
        inpsourcesim = 0
        for i in responseArr:
            simScore = 0   
            flag = 0
            for j in sourcearr:
                score = promptResponseSimilarity(j, i, headers)
                maxScore = max(maxScore,score)
                
                if flag == 0:
                    flag = 1
                    maxScore = max(maxScore, promptResponseSimilarity(j, response, headers))
                    score2 = promptResponseSimilarity(j, prompt, headers)
                    inpsourcesim = max(score2,inpsourcesim)
                if score > simScore:
                    simScore = score        
        
        if maxScore<0.3:            
            finalScore = round(1-(inpoutsim*0.2 + inpsourcesim*0.4 + maxScore*0.4).tolist(),2)
        elif maxScore>0.45:           
            finalScore=0.2
        else:         
            finalScore = round(1-(inpoutsim*0.2 + maxScore*0.8).tolist(),2)
        score = {"score":finalScore}
        return score

    except Exception as e:
            log.info("Failed at Show_Score")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

def identifyIDP(text):
    if 'IDP' in text:
        return True
    return False

def identifyEmoji(text):
    '''Function to find emojis in the text
        Args: string
        Return: dictionary'''
    emoji_values=demoji.findall(text)
    emoji_dict={}
    if len(emoji_values)>0:
        emoji_dict['flag']=True
    else:
        emoji_dict['flag']=False
    emoji_dict['value']=list(emoji_values.keys())
    emoji_dict['mean']=list(emoji_values.values())
    return emoji_dict

def emojiToText(text,emoji_dict):
    '''Function to convert emojis in a sentence to text
       Returns the modified text(text), text with emojis removed(privacy_text) and dictionary containing all emojis and their meanings present in input text(current_emoji_dict)'''
    emoji_list = sorted(emoji_data.keys(), key=len, reverse=True)
    current_emoji_dict=MultiValueDict()
    privacy_text=text
    #replacing emojis with their meaning from inappropriate_emoji.json
    for emoji in emoji_list:
        if emoji in text:
            pattern = regex.escape(emoji)
            occurrences = regex.findall(pattern, text, flags=regex.V1)
            text = text.replace(emoji, ' ' + emoji_data[emoji])
            privacy_text=privacy_text.replace(emoji,' ')
            for i in range(0,len(occurrences)):
                current_emoji_dict[emoji]=emoji_data[emoji]
            
    #replacing rest of the emojis with their meaning from emoji_dict
    for i in range(0,len(emoji_dict['value'])): 
            if emoji_dict['value'][i] in text:
                pattern = regex.escape(emoji_dict['value'][i])
                occurrences = regex.findall(pattern, text, flags=regex.V1)
                text=text.replace(emoji_dict['value'][i],(' '+emoji_dict['mean'][i]).replace('_',' '))
                privacy_text=privacy_text.replace(emoji_dict['value'][i],' ')
                for j in occurrences:
                    current_emoji_dict[j] = emoji_dict['mean'][emoji_dict['value'].index(j)]
    return text,privacy_text,current_emoji_dict

def wordToEmoji(text,current_emoji_dict,result):
    '''Function to check and convert profane word back to emoji(using it for profanity result)'''
    text1=text
    temp_dict=current_emoji_dict
    if len(result)>0:
        for i in range(0,len(result)):
            if result[i] not in text1:
                for j in list(temp_dict):
                    c=0
                    for k in temp_dict[j]:
                        if result[i] in k:
                            text1=text1.replace(result[i],'',1)
                            result[i]=j
                            temp_dict[j].pop(0)
                            c=1
                            break
                    if c==1:
                        break
            else:
                text1=text1.replace(result[i],'',1)
    return result

def profaneWordIndex(text,profane_list):
    '''Function to find location of profane words and emojis if emoji option is present in text'''
    index_list=[]
    for i in profane_list:
        if i in text:
            index_list.append([(text.find(i)),(text.find(i)+grapheme.length(str(i)))])
            alphabet_sequence = (string.ascii_lowercase * (grapheme.length(i) // 26 + 1))[:grapheme.length(i)]
            text=text.replace(i,alphabet_sequence,1)
    return index_list

#Custom dictionary class
class MultiValueDict(dict):
    def __setitem__(self, key, value):
        if key not in self:
            super().__setitem__(key, [])
        self[key].append(value)
    
    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        return self.get_all(key)
    
    def get_all(self, key):
        return super().__getitem__(key)       
