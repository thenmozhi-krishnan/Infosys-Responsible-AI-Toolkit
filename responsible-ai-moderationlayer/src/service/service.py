'''
Copyright 2024-2025 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import os
import re
import copy
import time
from config.logger import CustomLogger, request_id_var
import uuid
from datetime import datetime
import json
import requests
import asyncio
import threading
import openai
import numpy as np
import nltk
import traceback
import urllib3
from mapper.mapper import *
from dotenv import load_dotenv
from better_profanity import profanity
import aiohttp
import ssl
from smoothLLm import SMOOTHLLM
from telemetry import telemetry
from bergeron import  Bergeron
from dao.AdminDb import Results
from translate import ModelBasedTranslate
translator = ModelBasedTranslate()
from openai import AzureOpenAI
import demoji
import string 
import regex
import grapheme
import sys
from utilities.lruCaching import *
from utilities.utility_methods import *
import boto3
from botocore.exceptions import ClientError
import Llama_auth
import google.generativeai as genai
from fkscore import fkscore

log=CustomLogger()
request_id_var.set("Startup")
urllib3.disable_warnings()
startupFlag=True
global log_dict
log_dict={}
contentType = os.getenv("CONTENTTYPE")
aicloud_access_token=None
token_expiration=0

verify_ssl = os.getenv("VERIFY_SSL")
sslv={"False":False,"True":True,"None":True}

def handle_object(obj):
    return vars(obj) 

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


try:
    device = "cuda"
    load_dotenv()
    jailbreakurl=os.getenv("JAILBREAKMODEL")
    promptInjectionurl=os.getenv("PROMPTINJECTIONMODEL")
    detoxifyurl=os.getenv("DETOXIFYMODEL")
    mpnetsimilarityurl=os.getenv("SIMILARITYMODEL")
    topicurl=os.getenv("RESTRICTEDMODEL")
    privacyurl=os.getenv("PRIVACY")
    sentimenturl=os.getenv("SENTIMENT")
    invisibletexturl=os.getenv("INVISIBLETEXT")
    gibberishurl=os.getenv("GIBBERISH")
    bancodeurl=os.getenv("BANCODE")

    tel_env=os.getenv("TELEMETRY_ENVIRONMENT")
    telemetryurl = os.getenv("TELEMETRY_PATH") 
    coupledtelemetryurl=os.getenv("COUPLEDTELEMETRYPATH")
    EXE_CREATION = os.getenv("EXE_CREATION")
    cache_ttl = int(os.getenv("CACHE_TTL"))
    cache_size = int(os.getenv("CACHE_SIZE"))

    promptInjectionraiurl=os.getenv("PROMPTINJECTIONMODELRAI")
    jailbreakraiurl=os.getenv("JAILBREAKMODELRAI")
    detoxifyraiurl=os.getenv("DETOXIFYMODELRAI")
    topicraiurl=os.getenv("RESTRICTEDMODELRAI")
    mpnetsimilarityraiurl=os.getenv("SIMILARITYMODELRAI")
    privacyraiurl=os.getenv("PRIVACYRAI")

    target_env=os.getenv("TARGETENVIRONMENT")
   
    cache_flag = os.getenv("CACHE_FLAG")

    ## FOR NORMAL APP RUNNING
    if(EXE_CREATION == "True"):
        # Get the base path (this will be the path to the executable when running the bundled app)
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Construct the absolute path to the nltk_data directory
    nltk_path = os.path.join(base_path, 'data','nltk_data')
    nltk.data.path.append(nltk_path) # Append the nltk_path to the nltk data path
    
    jailbreak_embeddings_path = os.path.join(base_path, 'data/jailbreak_embeddings.json')
    refusal_embeddings_path = os.path.join(base_path, 'data/refusal_embeddings.json')
    topic_embeddings_path = os.path.join(base_path, 'data/topic_embeddings.json')
    orgpolicy_embeddings_path = os.path.join(base_path, 'data/orgpolicy_embeddings.json')
    inappropriate_emoji_path = os.path.join(base_path, 'data/inappropriate_emoji.json')
    moderation_time_json = os.path.join(base_path, 'data/moderationtime.json')

    with open(jailbreak_embeddings_path, "r") as file:
        json_data = file.read()
        jailbreak_embeddings = json.loads(json_data)
    with open(refusal_embeddings_path, "r") as file:
        json_data = file.read()
        refusal_embeddings = json.loads(json_data)
    with open(topic_embeddings_path, "r") as file:
        json_data = file.read()
        topic_embeddings = json.loads(json_data)
    with open(orgpolicy_embeddings_path, "r") as file:
        json_data = file.read()
        orgpolicy_embeddings = json.loads(json_data)
    
    #load the json file for imappropriate emojis defined
    with open(inappropriate_emoji_path,  encoding="utf-8",mode="r") as emoji_file:
        data=emoji_file.read()
        emoji_data=json.loads(data)
    


except Exception as e:
    log.error(str(traceback.extract_tb(e.__traceback__)[0].lineno))
    log.info(f"Exception: {e}")
    



async def post_request(url, data=None, json=None, headers=None, verify=sslv[verify_ssl]):
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
                    {"Prompt Injection Check": "0s", 
                    "Jailbreak Check": "0s", 
                    "Toxicity Check": "0s", 
                    "Privacy Check": "0s", 
                    "Profanity Check": "0s", 
                    "Refusal Check": "0s",
                    "Restricted Topic Check": "0s",
                    "Text Quality Check": "0s",
                    "Custom Theme Check": "0s",
                    "Random Noise Check": "0s", 
                    "Advanced Jailbreak Check": "0s"}, 
                "responseModeration": 
                    {"Toxicity Check": "0s", 
                    "Privacy Check": "0s", 
                    "Profanity Check": "0s", 
                    "Refusal Check": "0s", 
                    "Text Relevance Check": "0s", 
                    "Text Quality Check": "0s",
                    "Custom Theme Check": "0s"}, 
                "OpenAIInteractionTime": "0s",
                "translate":"0s",
                }

dictcheck={"Prompt Injection Check": "0s", 
           "Jailbreak Check": "0s", 
           "Toxicity Check": "0s", 
           "Privacy Check": "0s", 
           "Profanity Check": "0s", 
           "Refusal Check": "0s",
           "Restricted Topic Check": "0s",
           "Text Quality Check": "0s",
           "Custom Theme Check": "0s"}

moderation_timecheck = {}
# def writejson(dict_timecheck):            
#     json_object = json.dumps(dict_timecheck)
#     with open("data/moderationtime.json", "w") as outfile:
#         outfile.write(json_object)


# PATH MODIFIED FOR EXE
def writejson(dict_timecheck):            
    json_object = json.dumps(dict_timecheck)

    if(EXE_CREATION == "True"):
        # # Get the directory of the .exe file
        # exe_dir = os.path.dirname(sys.executable)
        # Create the path for the json file
        json_path = moderation_time_json
    else:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Create the path for the json file
        json_path = os.path.join(script_dir, "data/moderationtime.json")

    with open(json_path, "w") as outfile:
        outfile.write(json_object)



# PATH MODIFIED FOR EXE
def writeDecoupledTime(timecheck):            
    json_object = json.dumps(timecheck)

    if(EXE_CREATION == "True"):
        # # Get the directory of the .exe file
        # exe_dir = os.path.dirname(sys.executable)
        # Create the path for the json file
        json_path = os.path.join(base_path, "data/decoupledModerationtime.json")
    else:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Create the path for the json file
        json_path = os.path.join(script_dir, "data/decoupledModerationtime.json")

    with open(json_path, "w") as outfile:
        outfile.write(json_object)

###########################################

class PromptInjection:
    async def classify_text(self, text,headers):
        
        try:
            #response with azure moderation model endpoints
            if target_env=="azure":
                log.info("Using azure prompt injection model endpoint")
                output=await post_request(url=promptInjectionurl,json={"text": text},headers=headers)
                output=json.loads(output.decode('utf-8'))
                modeltime = output[2]["time_taken"]
                if output[0]=='LEGIT':
                    injectionscore = 1 - output[1]
                else:
                    injectionscore = output[1]
            #response with aicloud moderation model endpoints
            elif target_env=="aicloud":
                log.info("Using aicloud prompt injection model endpoint")
                st=time.time()
                output=await post_request(url=promptInjectionraiurl,json={"inputs": [text]},headers=headers)
                et=time.time()
                output=json.loads(output.decode('utf-8'))
                modeltime = str(round(st-et,3))+"s"
                if output[0]['label']=='LEGIT':
                    injectionscore = 1 - output[0]['score']
                else:
                    injectionscore = output[0]['score']

            return round(injectionscore,3),modeltime
        except Exception as e:
            log.error("Error occured in PromptInjection")
            line_number = traceback.extract_tb(e.__traceback__)[0].lineno
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at PromptInjection model call"})
            log.error(f"Exception: {line_number,e}")

class SentimentAnalysis:
    async def classify_text(self, text,headers):
        try:
            log.info("Using sentiment endpoint")
            output=await post_request(url=sentimenturl,json={"text": text},headers=headers)
            output=json.loads(output.decode('utf-8'))
            log.info(f"output : {output}")
            return output
        except Exception as e:
            log.error("Error occured in Sentiment Check")
            line_number = traceback.extract_tb(e.__traceback__)[0].lineno
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Sentiment Check"})
            log.error(f"Exception: {line_number,e}")



class InvisibleText:
    async def find_invisible_chars(self, text,banned_categories,headers):
        try:
            log.info("Using invisibletext endpoint")
            output=await post_request(url=invisibletexturl,json={"text": text,"banned_categories":banned_categories},headers=headers)
            output=json.loads(output.decode('utf-8'))
            log.info(f"output : {output}")
            return output
        except Exception as e:
            log.error("Error occured in Invisible Text Check")
            line_number = traceback.extract_tb(e.__traceback__)[0].lineno
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Invisible Text Check"})
            log.error(f"Exception: {line_number,e}")

class Gibberish:
    async def detect_gibberish(self, text,gibberish_labels,headers):
        try:
            log.info("Using gibberish endpoint")
            output=await post_request(url=gibberishurl,json={"text": text,"labels":gibberish_labels},headers=headers)
            output=json.loads(output.decode('utf-8'))
            log.info(f"output : {output}")
            return output
        except Exception as e:
            log.error("Error occured in gibberish Check")
            line_number = traceback.extract_tb(e.__traceback__)[0].lineno
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at gibberish Check"})
            log.error(f"Exception: {line_number,e}")


class BanCode:
    async def ban_code(self, text,headers):
        try:
            log.info("Using ban code endpoint")
            output=await post_request(url=bancodeurl,json={"text": text},headers=headers)
            output=json.loads(output.decode('utf-8'))
            log.info(f"output : {output}")
            return output
        except Exception as e:
            log.error("Error occured in BanCode Check")
            line_number = traceback.extract_tb(e.__traceback__)[0].lineno
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at BanCode Check"})
            log.error(f"Exception: {line_number,e}")


def text_quality(text):
    f = fkscore(text)
    ease_score = f.score['readability']
    grade_score = f.score['read_grade']
    return ease_score,grade_score

class promptResponse:
    async def promptResponseSimilarity (self,prompt,output_text,headers):
        try:
            if target_env=="azure":
                url=mpnetsimilarityurl
            elif target_env=="aicloud":
                url=mpnetsimilarityraiurl
            output =await post_request(url = url,json={"text1": prompt,"text2": output_text},headers=headers)
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
            #response with azure moderation model endpoints
            if target_env=='azure':
                log.info("Using azure jailbreak model endpoint")
                text_embedding =await post_request(url = jailbreakurl,json={"text": [text]},headers=headers)
                modelcalltime = json.loads(text_embedding.decode('utf-8'))[1]['time_taken']
                text_embedding=json.loads(text_embedding.decode('utf-8'))[0][0]
            #response with aicloud moderation model endpoints
            elif target_env=='aicloud':
                log.info("Using aicloud jailbreak model endpoint")
                st=time.time()
                text_embedding =await post_request(url = jailbreakraiurl,json={"inputs": [text]},headers=headers)
                et=time.time()
                modelcalltime = str(round(et-st,3))+"s"
                text_embedding=json.loads(text_embedding.decode('utf-8'))[0]           
            
            similarities = []
            for embedding in jailbreak_embeddings:
                dot_product = np.dot(text_embedding, embedding)
                norm_product = np.linalg.norm(text_embedding) * np.linalg.norm(embedding)
                similarity = round(dot_product / norm_product,4)
                similarities.append(similarity)
       
            return max(similarities),modelcalltime
        except Exception as e:
        
            log.error("Error occured in Jailbreak")
      
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at PromptInjection model call"})
       
            log.error(f"Exception: {e}")

class Customtheme:
    async def identify_jailbreak(self,text,headers,theme=None):
        try:
            theme.append(text)
            #response with azure moderation model endpoints
            if target_env=='azure':
                log.info("Using azure jailbreak model endpoint for custom theme")
                customTheme_embeddings =await post_request(url = jailbreakurl,json={"text": theme},headers=headers)
                customTheme_embeddings_decoded = json.loads(customTheme_embeddings.decode('utf-8'))
                modelcalltime = customTheme_embeddings_decoded[1]['time_taken']
                text_embedding=customTheme_embeddings_decoded[0]   
                # customTheme_embeddings=customTheme_embeddings_decoded[0][:-1]
                return text_embedding,modelcalltime
            #response with aicloud moderation model endpoints
            elif target_env=='aicloud':
                log.info("Using aicloud jailbreak model endpoint for custom theme")
                st=time.time()
                customTheme_embeddings =await post_request(url = jailbreakraiurl,json={"inputs": theme},headers=headers)
                et=time.time()
                customTheme_embeddings_decoded = json.loads(customTheme_embeddings.decode('utf-8'))
                modelcalltime = str(round(et-st,3))+"s"
                text_embedding=customTheme_embeddings_decoded  
                # customTheme_embeddings=customTheme_embeddings_decoded[:-1]
                return text_embedding,modelcalltime

            # similarities = []
            # for embedding in customTheme_embeddings:
            #     dot_product = np.dot(text_embedding, embedding)
            #     norm_product = np.linalg.norm(text_embedding) * np.linalg.norm(embedding)
            #     similarity = round(dot_product / norm_product,4)
            #     similarities.append(similarity)
            # return max(similarities),modelcalltime
        except Exception as e:
            log.error("Error occured in Customtheme")
         
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Customtheme"})
       
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
    

class CustomthemeRestricted:
    def identify_jailbreak(self, text,headers,theme=None):
        try:
            #Using azure jailbreak endpoint for custom theme restricted
            if target_env=='azure':
                log.info("Using azure jailbreak endpoint for custom theme restricted")
                text_embedding = requests.post(url = jailbreakurl,json={"text": [text]},headers=headers,verify=sslv[verify_ssl]).json()[0][0]
            #Using aicloud jailbreak endpoint for custom theme restricted
            elif target_env=='aicloud':
                log.info("Using aicloud jailbreak endpoint for custom theme restricted")
                text_embedding = requests.post(url = jailbreakraiurl,json={"inputs": [text]},headers=headers,verify=sslv[verify_ssl]).json()[0]
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
            #for response with azure moderation model endpoints
            if target_env == 'azure':
                log.info("Using azure jailbreak model endpoint for refusal check")
                text_embedding =await post_request(url = jailbreakurl,json={"text": [text]},headers=headers)
                text_embedding=json.loads(text_embedding.decode('utf-8'))[0][0]
            
            # for response with aicloud moderation model endpoints
            elif target_env == 'aicloud':
                log.info("Using aicloud jailbreak model endpoint for refusal check")
                text_embedding =await post_request(url = jailbreakraiurl,json={"inputs": [text]},headers=headers)
                text_embedding=json.loads(text_embedding.decode('utf-8'))[0]
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
    async def restrict_topic(self,text,config_details,headers,model="dberta"): 
        try:
            labels= config_details["ModerationCheckThresholds"]["RestrictedtopicDetails"]["Restrictedtopics"]
            #Using azure moderation model endpoint
            if target_env=='azure':
                log.info("Using the azure endpoint for restricted topic")
                output =await post_request(url = topicurl,json={"text": text,"labels":labels,"model":model},headers=headers)
                output=json.loads(output.decode('utf-8'))
                modelcalltime = output['time_taken']
                d={}
                for i in range(len(labels)):
                    d[output["labels"][i]] = str(round(output["scores"][i],3))

            #Using aicloud moderation model endpoint
            elif target_env=='aicloud':
                log.info("Using the aicloud model endpoint for restricted topic")
                st=time.time()
                output =await post_request(url = topicraiurl,json={"inputs": [{"text":text,"labels":labels}]},headers=headers)
                et=time.time()
                output=json.loads(output.decode('utf-8'))
                modelcalltime = str(round(et-st,3))+"s"
                d={}
                for i in range(len(labels)):
                    d[output[0]["labels"][i]] = str(round(output[0]["scores"][i],3))

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

                #Using azure moderation model endpoint 
                if target_env=='azure':
                    log.info("Using azure model endpoints for toxicity")
                    for chunk in chunked_texts:
                        result =await post_request(url=detoxifyurl,json={"text": chunk},headers=headers)
                        result=json.loads(result.decode('utf-8'))
                        modelcalltime = result["time_taken"]
                        toxicity_scoreslist.append(result)

                    for item in toxicity_scoreslist:
                        toxic_score_list = item['toxicScore']
                        for metric in toxic_score_list:
                            metric_name = metric['metricName']
                            metric_score = metric['metricScore']
                            toxicity_scores[metric_name] += metric_score
                        count += 1   

                #Using aicloud moderation model endpoint
                elif target_env=='aicloud':
                    log.info("Using aicloud model endpoints for toxicity")
                    for chunk in chunked_texts:
                        st=time.time()
                        result =await post_request(url=detoxifyraiurl,json={"inputs": [chunk]},headers=headers)
                        et=time.time()
                        result=json.loads(result.decode('utf-8'))
                        modelcalltime =str(round(et-st,3))+"s"
                        toxicity_scoreslist.append(result[0])

                    for item in toxicity_scoreslist:
                        for key,value in item.items():
                            metric_name = key
                            metric_score = value
                            toxicity_scores[metric_name] += metric_score
                        count += 1 
                toxicity_averages = {key: value / count for key, value in toxicity_scores.items()}
                
                l=[]
                for i in toxicity_averages:
                    l.append({'metricName': i, 'metricScore': toxicity_averages[i]})
                output = {"toxicScore":l}
                toxic_score = output["toxicScore"][0]["metricScore"]
    
            else:
                #Using azure moderation model endpoint 
                if target_env=='azure':
                    log.info("Using azure model endpoints for toxicity")
                    output = await post_request(url=detoxifyurl,json={"text": text},headers=headers,verify=False)
                    output=json.loads(output.decode('utf-8'))
                    modelcalltime = output["time_taken"]
                    toxic_score = output["toxicScore"][0]["metricScore"]

                #Using aicloud moderation model endpoint
                elif target_env=='aicloud':
                    log.info("Using aicloud model endpoints for toxicity")
                    st=time.time()
                    output=await post_request(url=detoxifyraiurl,json={"inputs":[text]},headers=headers,verify=False)
                    et=time.time()
                    output=json.loads(output.decode('utf-8'))
                    modelcalltime=str(round(et-st,3))+"s"
                    toxic_score = output[0]["toxicity"]
                    lst=[]
                    for key, value in output[0].items():
                        lst.append({"metricName":key,"metricScore":value})
                    output={'time_taken':modelcalltime,'toxicScore':lst}
            return toxic_score,output,modelcalltime
        except Exception as e:
        
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Toxicity model call"})
         
            log.error("Error occured in Toxicity")
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
            #Using azure endpoints for profanity popup
            if target_env=='azure':
                log.info("Using azure endpoints for profanity popup")
                for chunk in chunks:
                    result = requests.post(url=detoxifyurl,json={"text": chunk},headers=headers,verify=sslv[verify_ssl]).json()
                    toxicity_scoreslist.append(result)
                for item in toxicity_scoreslist:
                    toxic_score_list = item['toxicScore']
                    for metric in toxic_score_list:
                        metric_name = metric['metricName']
                        metric_score = metric['metricScore']
                        toxicity_scores[metric_name] += metric_score
                    count += 1
            #Using aicloud endpoints for profanity popup
            elif target_env=='aicloud':
                log.info("Using aicloud endpoints for profanity popup")
                for chunk in chunks:
                    result = requests.post(url=detoxifyraiurl,json={"inputs": [chunk]},headers=headers,verify=sslv[verify_ssl]).json()
                    toxicity_scoreslist.append(result[0])
                for item in toxicity_scoreslist:
                        for key,value in item.items():
                            metric_name = key
                            metric_score = value
                            toxicity_scores[metric_name] += metric_score
                        count += 1 
           
            toxicity_averages = {key: value / count for key, value in toxicity_scores.items()}
            
            l=[]
            for i in toxicity_averages:
                l.append({'metricName': i, 'metricScore': toxicity_averages[i]})
            output = {"toxicScore":l}
            toxic_score = output["toxicScore"][0]["metricScore"]
        else:
            #Using azure endpoints for profanity popup
            if target_env=='azure':
                log.info("Using azure endpoints for profanity popup")
                output = requests.post(url=detoxifyurl,json={"text": text},headers=headers,verify=sslv[verify_ssl]).json()
                toxic_score = output["toxicScore"][0]["metricScore"]
            #Using aicloud endpoints for profanity popup
            elif target_env=='aicloud':
                log.info("Using aicloud endpoints for profanity popup")
                output = requests.post(url=detoxifyraiurl,json={"inputs": [text]},headers=headers,verify=sslv[verify_ssl]).json()
                toxic_score = output[0]["toxicity"]              


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
            if emoji_flag:
                privacy_text=emojiToText(text=text,emoji_dict=emoji_dict)[1]
        text=privacy_text if emoji_flag else text
        if target_env=='azure':
            url=privacyurl
        elif target_env=='aicloud':
            url=privacyraiurl
        analyze_result =requests.post(url=url,json={"text": text},headers=headers)
        analyze_result=analyze_result.json()
        entitiesconfigured = payload.PiientitiesConfiguredToDetect
        entitiesconfiguredToBlock = payload.PiientitiesConfiguredToBlock

        entityList=[]   
        result = "Passed"        
        for i in range(0,len(analyze_result["PIIresult"])):

            if analyze_result["PIIresult"][i]["type"] in entitiesconfiguredToBlock and analyze_result["PIIresult"][i]["score"]>0.4:
                result="Block"
                entity_obj = PiiEntitiesforPopup(EntityType = analyze_result["PIIresult"][i]["type"] ,
                                        beginOffset = analyze_result["PIIresult"][i]["beginOffset"],
                                        endOffset = analyze_result["PIIresult"][i]["endOffset"],
                                        score= analyze_result["PIIresult"][i]["score"],
                                        value = analyze_result["PIIresult"][i]["responseText"]) 
                entityList.append(entity_obj)

            if analyze_result["PIIresult"][i]["type"] in entitiesconfigured and analyze_result["PIIresult"][i]["score"]>0.4 and analyze_result["PIIresult"][i]["type"] not in entitiesconfiguredToBlock:
                entity_obj = PiiEntitiesforPopup(EntityType = analyze_result["PIIresult"][i]["type"] ,
                                        beginOffset = analyze_result["PIIresult"][i]["beginOffset"],
                                        endOffset = analyze_result["PIIresult"][i]["endOffset"],
                                        score= analyze_result["PIIresult"][i]["score"],
                                        value = analyze_result["PIIresult"][i]["responseText"]) 
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

                #Using azure moderation model endpoint 
                if target_env=='azure':
                    log.info("Using azure model endpoints for profanity")
                    for chunk in chunked_texts:
                        result =await post_request(url=detoxifyurl,json={"text": chunk},headers=headers)
                        result=json.loads(result.decode('utf-8'))
                    toxicity_scoreslist.append(result)

                    for item in toxicity_scoreslist:
                        toxic_score_list = item['toxicScore']
                        for metric in toxic_score_list:
                            metric_name = metric['metricName']
                            metric_score = metric['metricScore']
                            toxicity_scores[metric_name] += metric_score
                        count += 1   

                #Using aicloud moderation model endpoint
                elif target_env=='aicloud':
                    log.info("Using aicloud model endpoints for profanity")
                    for chunk in chunked_texts:
                        result =await post_request(url=detoxifyraiurl,json={"inputs": [chunk]},headers=headers)
                        result=json.loads(result.decode('utf-8'))
                    toxicity_scoreslist.append(result[0])

                    for item in toxicity_scoreslist:
                        for key,value in item.items():
                            metric_name = key
                            metric_score = value
                            toxicity_scores[metric_name] += metric_score
                        count += 1 
                
                toxicity_averages = {key: value / count for key, value in toxicity_scores.items()}
                
                l=[]
                for i in toxicity_averages:
                    l.append({'metricName': i, 'metricScore': toxicity_averages[i]})
                output = {"toxicScore":l}
                toxic_score = output["toxicScore"][0]["metricScore"]
            else:
                #Using azure moderation model endpoint 
                if target_env=='azure':
                    log.info("Using azure model endpoints for toxicity")
                    output = await post_request(url=detoxifyurl,json={"text": text},headers=headers,verify=False)
                    output=json.loads(output.decode('utf-8'))
                    toxic_score = output["toxicScore"][0]["metricScore"]

                #Using aicloud moderation model endpoint
                elif target_env=='aicloud':
                    log.info("Using aicloud model endpoints for toxicity")
                    st=time.time()
                    output=await post_request(url=detoxifyraiurl,json={"inputs":[text]},headers=headers,verify=False)
                    et=time.time()
                    output=json.loads(output.decode('utf-8'))
                    toxic_score = output[0]["toxicity"]
            if toxic_score > 0.6:
                res = profanity.censor(text)
                return res[1]
            else:
                return []
        except Exception as e:
            log.error("Error occured in Profanity")
          
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Profanity"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            
#PII 
class PII:
    async def analyze(self,text,headers):
        try:
            if target_env=='azure':
                url=privacyurl
            elif target_env=='aicloud':
                url=privacyraiurl
            analyze_result =await post_request(url=url,json={"text": text},headers=headers)
            analyze_result=json.loads(analyze_result.decode('utf-8'))
            modelcalltime = analyze_result["modelcalltime"]
            entityDict= {}
            entityDict["types"] = [dict["type"] for dict in analyze_result["PIIresult"]]
            entityDict["scores"] = [dict["score"] for dict in analyze_result["PIIresult"]]
            entityDict["responseTexts"]=[dict["responseText"] for dict in analyze_result["PIIresult"]]
            return entityDict,modelcalltime
        except Exception as e:
            log.error("Error occured in PII")
          
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at PII call"})
          
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
        self.ToxicityThreshold = (None if config_details['ModerationCheckThresholds'].get('ToxicityThresholds')==None else config_details['ModerationCheckThresholds']['ToxicityThreshold']["ToxicityThreshold"])
        self.RefusalThreshold = config_details["ModerationCheckThresholds"].get('RefusalThreshold')
        self.PIIenities_selectedToBlock = config_details['ModerationCheckThresholds'].get('PiientitiesConfiguredToBlock')
        self.Topic_threshold = (None if config_details['ModerationCheckThresholds'].get("RestrictedtopicDetails")==None else config_details['ModerationCheckThresholds']["RestrictedtopicDetails"]['RestrictedtopicThreshold'])
        self.SmoothLT=config_details['ModerationCheckThresholds'].get('SmoothLlmThreshold')   # added for smoothllm
        
        self.sentiment_threshold=(None if config_details['ModerationCheckThresholds'].get('SentimentThreshold')==None else config_details['ModerationCheckThresholds']['SentimentThreshold'])
        self.invisibletext_threshold=None
        self.invisibletext_categories=None
        self.gibberish_threshold=None
        self.gibberish_labels=None
        if config_details['ModerationCheckThresholds'].get('InvisibleTextCountDetails')!=None:
            self.invisibletext_threshold=config_details['ModerationCheckThresholds'].get('InvisibleTextCountDetails')['InvisibleTextCountThreshold']
            self.invisibletext_categories=config_details['ModerationCheckThresholds'].get('InvisibleTextCountDetails')['BannedCategories']
        if config_details['ModerationCheckThresholds'].get('GibberishDetails')!=None:
            self.gibberish_threshold=config_details['ModerationCheckThresholds'].get('GibberishDetails')['GibberishThreshold']
            self.gibberish_labels=config_details['ModerationCheckThresholds'].get('GibberishDetails')['GibberishLabels']
        self.bancode_threshold=(None if config_details['ModerationCheckThresholds'].get('BanCodeThreshold')==None else config_details['ModerationCheckThresholds']['BanCodeThreshold'])
        
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
        self.dict_sentiment={}
        self.dict_invisibleText={}
        self.dict_gibberish={}
        self.dict_bancode={}
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
        self.dict_prompt['object']=promptInjectionCheck(injectionConfidenceScore = str("),
                                          injectionThreshold = str(""),
                                          result = 'UNMODERATED')
        self.dict_jailbreak['object']=jailbreakCheck(jailbreakSimilarityScore = str('),
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
        self.dict_sentiment['object']= sentimentCheck(score = str("),
                                          threshold = str(""),
                                          result = 'UNMODERATED')
        self.dict_invisibleText['object']= invisibleTextCheck(invisibleTextIdentified = [],
                                          threshold = str(""),
                                          result = 'UNMODERATED')
        self.dict_gibberish['object']= gibberishCheck(gibberishScore = [],
                                          threshold = str(""),
                                          result = 'UNMODERATED')
        self.dict_bancode['object']= bancodeCheck(score = [],
                                          threshold = str(""),
                                          result = 'UNMODERATED')      

    
    async def validate_sentiment(self,headers):
        try:
            log.info(f"Initialising Sentiment validation")
            self.dict_sentiment['key'] = "Sentiment Check"
            if self.sentiment_threshold==None:
                self.dict_sentiment['status'] = True
                return [self.dict_sentiment]
            
            st = time.time()
            log.info(f"threshold : {str(round(float(self.sentiment_threshold),2))}")
            sentiment_check = SentimentAnalysis()
            output = await sentiment_check.classify_text(self.text,headers)
            sentiment_score = output['score']['compound']
            self.modeltime["Sentiment Check"]=output['time_taken']
            
            if sentiment_score < self.sentiment_threshold:
                obj_sentiment = sentimentCheck(score = str(round(float(sentiment_score),2)),
                                            threshold = str(round(float(self.sentiment_threshold),2)),
                                            result = 'FAILED')
                self.dict_sentiment['status'] = False
            else:
                obj_sentiment = sentimentCheck(score = str(sentiment_score),
                                            threshold = str(self.sentiment_threshold),
                                            result = 'PASSED')
                self.dict_sentiment['status'] = True

            self.dict_sentiment['object'] = obj_sentiment
            et = time.time()
            rt = et - st
            dictcheck["Sentiment Check"]=str(round(rt,3))+"s"
            log.debug(f"Sentiment run time: {rt}")
            self.timecheck["Sentiment Check"]=str(round(rt,3))+"s"
            
            return [self.dict_sentiment]
        except Exception as e:
            log.error("Failed at sentiment_check")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at sentiment_check"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

    
    async def validate_invisibletext(self,headers):
        try:
            log.info(f"Initialising Invisible Text validation")
            self.dict_invisibleText['key'] = "Invisible Text Check"
            if self.invisibletext_threshold==None:
                self.dict_invisibleText['status'] = True
                return [self.dict_invisibleText]
            
            st = time.time()
            invisibletext_check = InvisibleText()
            output = await invisibletext_check.find_invisible_chars(self.text,self.invisibletext_categories,headers)
            invisiblecharsFound = output['result']
            self.modeltime["Invisible Text Check"]=output['time_taken']
            log.info(f"threshold : {str(self.invisibletext_threshold)}")
            if len(invisiblecharsFound) >= self.invisibletext_threshold:
                obj_invisibletext = invisibleTextCheck(invisibleTextIdentified=invisiblecharsFound,
                                            threshold = str(self.invisibletext_threshold),
                                            result = 'FAILED')
                self.dict_invisibleText['status'] = False
            else:
                obj_invisibletext = invisibleTextCheck(invisibleTextIdentified=invisiblecharsFound,
                                            threshold = str(self.invisibletext_threshold),
                                            result = 'PASSED')
                self.dict_invisibleText['status'] = True

            self.dict_invisibleText['object'] = obj_invisibletext
            et = time.time()
            rt = et - st
            dictcheck["Invisible Text Check"]=str(round(rt,3))+"s"
            log.debug(f"Invisible text run time: {rt}")
            self.timecheck["Invisible Text Check"]=str(round(rt,3))+"s"
            
            return [self.dict_invisibleText]
        except Exception as e:
            log.error("Failed at invisibletext_check")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at invisibletext_check"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

    async def validate_gibberish(self,headers):
        try:
            self.dict_gibberish['key'] = "Gibberish Check"
            if self.gibberish_threshold==None:
                self.dict_gibberish['status'] = True
                return [self.dict_gibberish]

            log.info(f"Initialising Gibberish validation")
            st = time.time()
            gibberish_check = Gibberish()
            output = await gibberish_check.detect_gibberish(self.text,self.gibberish_labels,headers)
            log.info(f"threshold : {str(self.gibberish_threshold)}")
            for i in range(len(output['result'])):
                gibberish_score = output['result'][i]['gibberish_score']
                if gibberish_score > self.gibberish_threshold and output['result'][i]['gibberish_label']=="noise":
                    obj_gibberish = gibberishCheck(gibberishScore=output['result'],
                                                threshold = str(self.gibberish_threshold),
                                                result = 'FAILED')
                    self.dict_gibberish['status'] = False
                    break
              
                obj_gibberish = gibberishCheck(gibberishScore=output['result'],
                                                threshold = str(self.gibberish_threshold),
                                                result = 'PASSED')
                self.dict_gibberish['status'] = True

            self.modeltime["Gibberish Check"]=output['time_taken']
            self.dict_gibberish['object'] = obj_gibberish
            et = time.time()
            rt = et - st
            dictcheck["Gibberish Check"]=str(round(rt,3))+"s"
            log.debug(f"Gibberish text run time: {rt}")
            self.timecheck["Gibberish Check"]=str(round(rt,3))+"s"
            
            return [self.dict_gibberish]
        except Exception as e:
            log.error("Failed at Gibberish_check")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Gibberish_check"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

    
    async def validate_bancode(self,headers):
        try:
            self.dict_bancode['key'] = "Ban Code Check"
            log.info(f"Initialising Ban Code validation")
            st = time.time()
            bancode_check = BanCode()
            output = await bancode_check.ban_code(self.text,headers)
            res = output['result']
            if res['label']=="CODE":
                obj_bancode = bancodeCheck(label=res['label'],result = 'FAILED')
                self.dict_bancode['status'] = False 
            else: 
                obj_bancode = bancodeCheck(label=res['label'],result = 'PASSED')
                self.dict_bancode['status'] = True

            self.modeltime["Ban Code Check"]=output['time_taken']
            self.dict_bancode['object'] = obj_bancode
            et = time.time()
            rt = et - st
            dictcheck["Ban Code Check"]=str(round(rt,3))+"s"
            log.debug(f"Ban Code Check run time: {rt}")
            self.timecheck["Ban Code Check"]=str(round(rt,3))+"s"
            
            return [self.dict_bancode]
        except Exception as e:
            log.error("Failed at bancode_check")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at bancode_check"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")


    async def validate_smoothllm(self,headers):
        try:
            log.info(f"Initialising smoothllm validation")
            st = time.time()        
            #emoji check
            if self.emoji_flag:
                threshold, defense_output =  SMOOTHLLM.main(self.deployment_name,self.privacy_text, self.SmoothLT['input_pertubation'], self.SmoothLT['number_of_iteration'])
            else:
                threshold, defense_output =  SMOOTHLLM.main(self.deployment_name,self.text, self.SmoothLT['input_pertubation'], self.SmoothLT['number_of_iteration'])

            self.dict_smoothllm['key'] = 'Random Noise Check'
            
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
                    dictcheck["Random Noise Check"]=str(round(rt,3))+"s"
                    log.info(f"smoothllm run time: {rt}")
                    
                    return [self.dict_smoothllm]
                
                
            if threshold >= self.SmoothLT['SmoothLlmThreshold']:
                obj_smooth = smoothLlmCheck(
                                            smoothLlmScore = str(threshold),
                                            smoothLlmThreshold= str(self.SmoothLT['SmoothLlmThreshold']),
                                            result='FAILED')
    
                self.dict_smoothllm['object'] = obj_smooth
                self.dict_smoothllm['status'] = False
                et = time.time()
                rt = et - st
                dictcheck["Random Noise Check"]=str(round(rt,3))+"s"
                log.info(f"smoothllm run time: {rt}")
                
                return [self.dict_smoothllm]
            elif threshold == -1:
                obj_smooth = smoothLlmCheck(smoothLlmScore = str(threshold),
                                            smoothLlmThreshold= str(self.SmoothLT['SmoothLlmThreshold']),
                                            result = 'UNDETERMINED')
                self.dict_smoothllm['object'] = obj_smooth
                self.dict_smoothllm['status'] = False
                et = time.time()
                rt = et - st
                dictcheck["Random Noise Check"]=str(round(rt,3))+"s"
                log.info(f"smoothllm run time: {rt}")
                
                return [self.dict_smoothllm]
            else:
                obj_smooth = smoothLlmCheck(smoothLlmScore = str(threshold),
                                            smoothLlmThreshold= str(self.SmoothLT['SmoothLlmThreshold']),
                                            result = 'PASSED')
                self.dict_smoothllm['object'] = obj_smooth
                self.dict_smoothllm['status'] = True
                et = time.time()
                rt = et - st
                dictcheck["Random Noise Check"]=str(round(rt,3))+"s"
                log.info(f"Smoothllm run time: {rt}")
             
                return [self.dict_smoothllm]
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
            
            self.dict_bergeron['key'] = 'Advanced Jailbreak Check'
            
            if flag == "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766":
                obj_berger = bergeronCheck(
                                            text="UNDETERMINED",                                            
                                            result='PASSED')
    
                self.dict_bergeron['object'] = obj_berger
                self.dict_bergeron['status'] = True
                et = time.time()
                rt = et - st
                dictcheck["Advanced Jailbreak Check"]=str(round(rt,3))+"s"
                log.info(f"Bergeron run time: {rt}")
                
                return [self.dict_bergeron]
            
            if flag == "FAILED":
                obj_berger = bergeronCheck(
                                            text="ADVERSARIAL",                                            
                                            result='FAILED')
    
                self.dict_bergeron['object'] = obj_berger
                self.dict_bergeron['status'] = False
                et = time.time()
                rt = et - st
                dictcheck["Advanced Jailbreak Check"]=str(round(rt,3))+"s"
                log.info(f"Bergeron run time: {rt}")
                
                return [self.dict_bergeron]
            
            elif flag == "UNDETERMINED":
                obj_berger = bergeronCheck(
                                            text="Cannot be determined as AWS Creds expired",                                            
                                            result='UNDETERMINED')
    
                self.dict_bergeron['object'] = obj_berger
                self.dict_bergeron['status'] = False
                et = time.time()
                rt = et - st
                dictcheck["Advanced Jailbreak Check"]=str(round(rt,3))+"s"
                log.info(f"Bergeron run time: {rt}")
                
                return [self.dict_bergeron]
            
            else:
                obj_berger = bergeronCheck(
                                            text="NON ADVERSARIAL",
                                            result = 'PASSED')
                self.dict_bergeron['object'] = obj_berger
                self.dict_bergeron['status'] = True
                et = time.time()
                rt = et - st
                dictcheck["Advanced Jailbreak Check"]=str(round(rt,3))+"s"
                log.info(f"Bergeron run time: {rt}")                
             
                return [self.dict_bergeron]
            
            
        except Exception as e:
            log.error("Failed at validate_bergeron")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at validate_bergeron"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            
    async def validate_prompt(self,headers):
        try:
            log.info(f"Initialising PromptInjection validation")
            st = time.time()
            prompt_check = PromptInjection()
            injectionscore, modelcalltime = await prompt_check.classify_text(self.text,headers)
            self.modeltime["Prompt Injection Check"]=modelcalltime
            self.dict_prompt['key'] = 'Prompt Injection Check'
            if injectionscore >= self.promptInjection_threshold:
                obj_prompt = promptInjectionCheck(injectionConfidenceScore = str(round(injectionscore,2)),
                                            injectionThreshold = str(self.promptInjection_threshold),
                                            result = 'FAILED')
                self.dict_prompt['status'] = False
            else:
                obj_prompt = promptInjectionCheck(injectionConfidenceScore = str(injectionscore),
                                            injectionThreshold = str(self.promptInjection_threshold),
                                            result = 'PASSED')
                self.dict_prompt['status'] = True

            self.dict_prompt['object'] = obj_prompt
            et = time.time()
            rt = et - st
            dictcheck["Prompt Injection Check"]=str(round(rt,3))+"s"
            log.debug(f"PromptInjection run time: {rt}")
            self.timecheck["Prompt Injection Check"]=str(round(rt,3))+"s"
            
            return [self.dict_prompt]
        except Exception as e:
            log.error("Failed at validate_prompt")
           
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at validate_prompt"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            
            
    # async def validate_jailbreak(self,headers):
    #     try:
    #         log.info(f"Initialising jailbreak validation")
    #         st = time.time()
    #         jailbreak = Jailbreak()
    #         result, modelcalltime = await jailbreak.identify_jailbreak(self.text, headers)
    #         self.modelcall['Jailbreak Check'] = modelcalltime
    #         self.dict_jailbreak['key'] = 'Jailbreak Check'
    #         if result <= self.Jailbreak_threshold:
    #             obj_jailbreak = jailbreakCheck(jailbreakSimilarityScore = str(round(float(result),2)),
    #                                         jailbreakThreshold = str(self.Jailbreak_threshold),
    #                                         result = 'PASSED')
    #             self.dict_jailbreak['status'] = True
    #         else:
    #             obj_jailbreak = jailbreakCheck(jailbreakSimilarityScore =  str(round(float(result),2)),
    #                                         jailbreakThreshold = str(self.Jailbreak_threshold),
    #                                         result = 'FAILED')
    #             self.dict_jailbreak['status'] = False

    #         self.dict_jailbreak['object'] = obj_jailbreak
    #         et = time.time()
    #         rt = et - st
    #         dictcheck["Jailbreak Check"]=str(round(rt,3))+"s"
    #         log.debug(f"jailbreak run time: {rt}")
    #         self.timecheck["Jailbreak Check"]=str(round(rt,3))+"s"
                
    #         return self.dict_jailbreak
    #     except Exception as e:
    #         log.error("Failed at validate jailbreak")
          
    #         log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
    #                                                "Error Module":"Failed at validate jailbreak"})

    #         # log.error(f"Exception: {e}")
    #         log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

    # async def validate_customtheme(self,theme,headers):
    #     try:
    #         log.info(f"Initialising Customtheme validation")
    #         st = time.time()
    #         customtheme = Customtheme()
    #         result, modelcalltime = await customtheme.identify_jailbreak(self.text,headers,theme.ThemeTexts)
    #         self.modeltime["Custom Theme Check"]=modelcalltime
    #         self.dict_customtheme['key'] = 'Custom Theme Check'
    #         if result <= theme.Themethresold:
    #             obj_jailbreak = customThemeCheck(customSimilarityScore = str(round(float(result),2)),
    #                                         themeThreshold = str(theme.Themethresold),
    #                                         result = 'PASSED')
    #             self.dict_customtheme['status'] = True 
    #         else:
    #             obj_jailbreak = customThemeCheck(customSimilarityScore =  str(round(float(result),2)),
    #                                         themeThreshold = str(theme.Themethresold),
    #                                         result = 'FAILED')
    #             self.dict_customtheme['status'] = False
            
    #         self.dict_customtheme['object'] = obj_jailbreak
    #         et = time.time()
    #         rt = et - st
    #         dictcheck["Custom Theme Check"]=str(round(rt,3))+"s"
    #         log.debug(f"CustomTheme run time: {rt}")
    #         self.timecheck["Custom Theme Check"]=str(round(rt,3))+"s"
    #         return self.dict_customtheme
        
    #     except Exception as e:
    #         log.error("Failed at validate customtheme")

    #         log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
    #                                                "Error Module":"Failed at validate customtheme"})
    #         log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")


    def jailbreak_val(self,text_embedding,modelcalltime,st,checkRes):
                    #print("---------------------------InsideJailbreak---------------------------")
                    similarities = []
                    # st=time.time()
                    for embedding in jailbreak_embeddings:
                        # similarity = requests.post(url = mpnetsimilarityurl,json={"emb1":text_embedding,"emb2":embedding},verify=False).json()[0][0]
                        # similarity = util.pytorch_cos_sim(text_embedding, embedding)
                        dot_product = np.dot(text_embedding, embedding)
                        norm_product = np.linalg.norm(text_embedding) * np.linalg.norm(embedding)
                        similarity = round(dot_product / norm_product,4)
                        similarities.append(similarity)
                    result = max(similarities)
                    # self.modelcall["Jailbreak Check"]=mt
                    self.modeltime['Jailbreak Check'] = modelcalltime
                    self.dict_jailbreak['key'] = 'Jailbreak Check'
                    if result <= self.Jailbreak_threshold:
                        obj_jailbreak = jailbreakCheck(jailbreakSimilarityScore = str(round(float(result),2)),
                                                    jailbreakThreshold = str(self.Jailbreak_threshold),
                                                    result = 'PASSED')
                        self.dict_jailbreak['status'] = True
                    else:
                        obj_jailbreak = jailbreakCheck(jailbreakSimilarityScore =  str(round(float(result),2)),
                                                    jailbreakThreshold = str(self.Jailbreak_threshold),
                                                    result = 'FAILED')
                        self.dict_jailbreak['status'] = False

                    self.dict_jailbreak['object'] = obj_jailbreak
                    et = time.time()
                    rt = et - st
                    dictcheck["Jailbreak Check"]=str(round(rt,3))+"s"
                    log.debug(f"jailbreak run time: {rt}")
                    self.timecheck["Jailbreak Check"]=str(round(rt,3))+"s"
                    checkRes.append(self.dict_jailbreak)
    
    def refusal_val(self,text_embedding,modelcalltime,st,checkRes):
                    #print("---------------------------InsideRefusal---------------------------")
                    similarities = []
                    for embedding in refusal_embeddings:
                        dot_product = np.dot(text_embedding, embedding)
                        norm_product = np.linalg.norm(text_embedding) * np.linalg.norm(embedding)
                        similarity = round(dot_product / norm_product,4)
                        # similarity = util.pytorch_cos_sim(text_embedding, embedding)
                        # similarity = requests.post(url = mpnetsimilarityurl,json={"emb1":text_embedding,"emb2":embedding},verify=False).json()[0][0]
                        similarities.append(similarity)
                    result = max(similarities)
                    # self.modelcall["Refusal Check"]=mt
                    self.dict_refusal['key'] = 'Refusal Check'
                    if result <= self.RefusalThreshold:
                        obj_refusal= refusalCheck(refusalSimilarityScore = str(round(float(result),2)),
                                                    RefusalThreshold = str(self.RefusalThreshold),
                                                    result = 'PASSED')
                        self.dict_refusal['status'] = True
                    else:
                        obj_refusal = refusalCheck(refusalSimilarityScore =  str(round(float(result),2)),
                                                    RefusalThreshold = str(self.RefusalThreshold),
                                                    result = 'FAILED')
                        self.dict_refusal['status'] = False

                    self.dict_refusal['object'] = obj_refusal
                    et = time.time()
                    rt = et - st
                    dictcheck["Refusal Check"]=str(round(rt,3))+"s"
                    log.debug(f"refusal run time: {rt}")
                    self.timecheck["Refusal Check"]=str(round(rt,3))+"s"
                    checkRes.append(self.dict_refusal)
        
    def custome_val(self,theme,customTheme_embeddings,text_embedding,modelcalltime,st,checkRes):
                    #print("---------------------------InsideCustomizedTheme---------------------------")
                    similarities = []

                    for embedding in customTheme_embeddings:
                        # similarity = requests.post(url = mpnetsimilarityurl,json={"emb1":text_embedding,"emb2":embedding},verify=False).json()[0][0]
                        # similarity = util.pytorch_cos_sim(text_embedding, embedding)
                        dot_product = np.dot(text_embedding, embedding)
                        norm_product = np.linalg.norm(text_embedding) * np.linalg.norm(embedding)
                        similarity = round(dot_product / norm_product,4)
                        similarities.append(similarity)
                    result=0
                    if(len(similarities)!=0):
                        result=max(similarities)
                    self.modeltime["Custom Theme Check"]=modelcalltime
                    # self.modelcall["Custom Theme Check"]=mt
                    self.dict_customtheme['key'] = 'Custom Theme Check'
                    if result <= theme.Themethresold:
                        obj_jailbreak = customThemeCheck(customSimilarityScore = str(round(float(result),2)),
                                                    themeThreshold = str(theme.Themethresold),
                                                    result = 'PASSED')
                        self.dict_customtheme['status'] = True 
                    else:
                        obj_jailbreak = customThemeCheck(customSimilarityScore =  str(round(float(result),2)),
                                                    themeThreshold = str(theme.Themethresold),
                                                    result = 'FAILED')
                        self.dict_customtheme['status'] = False

                    self.dict_customtheme['object'] = obj_jailbreak
                    et = time.time()
                    rt = et - st
                    dictcheck["Custom Theme Check"]=str(round(rt,3))+"s"
                    log.debug(f"CustomTheme run time: {rt}")
                    self.timecheck["Custom Theme Check"]=str(round(rt,3))+"s"
                    checkRes.append(self.dict_customtheme)
 
    async def validate_customtheme(self,theme,headers):
        try:
            log.info(f"Initialising Customtheme validation")
            st = time.time()
            customtheme = Customtheme()
            #print("Theam----",theme)
            results, modelcalltime= await customtheme.identify_jailbreak(self.text,headers,theme.ThemeTexts)
            checkRes=[]
            text_embedding=results[-1]
            threads=[]
            if("JailBreak" in self.Checks_selected):
                    #print("---------------------------Jailbreak---------------------------")
                    thread=threading.Thread(target=self.jailbreak_val,args=(text_embedding,modelcalltime,st,checkRes))
                    thread.start()
                    threads.append(thread)
                    # threads.append(threading.Thread(target=self.jailbreak_val,args=(text_embedding,modelcalltime,mt,st,checkRes)))
            if("Refusal" in self.Checks_selected):
                    #print("---------------------------Refusal---------------------------")
                    thread=threading.Thread(target=self.refusal_val,args=(text_embedding,modelcalltime,st,checkRes))
                    thread.start()
                    threads.append(thread)
                    # threads.append(threading.Thread(target=self.refusal_val,args=(text_embedding,modelcalltime,mt,st,checkRes)))
            if("CustomizedTheme" in self.Checks_selected):
                    #print("---------------------------CustomizedTheme---------------------------")
                
                    customTheme_embeddings=results[:-1]
                    thread=threading.Thread(target=self.custome_val,args=(theme,customTheme_embeddings,text_embedding,modelcalltime,st,checkRes))
                    thread.start()
                    threads.append(thread)
                    # threads.append(threading.Thread(target=self.custome_val,args=(theme,customTheme_embeddings,text_embedding,modelcalltime,mt,st,checkRes)))
                    # #print("customTheme_embeddings",len(customTheme_embeddings))

        

            for thread in threads:
                thread.join()       
                    # return [self.dict_customtheme]
            return checkRes
        except Exception as e:
            log.error("Failed at validate customtheme")

            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at validate customtheme"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")


    # async def validate_profanity(self):
    #     try:
    #         log.info(f"Initialising profanity validation")
    #         st = time.time()
    #         profanity = Profanity()
    #         #check emoji
    #         if self.emoji_flag:
    #             result = await profanity.recognise(self.converted_text)
    #             #check and convert profane word back to emoji
    #             result=wordToEmoji(self.text,self.current_emoji_dict,result)
                
    #         else:
    #             result = await profanity.recognise(self.text)
    #         self.dict_profanity['key'] = 'Profanity Check'
    #         if len(result) < self.Profanity_threshold:
    #             obj_profanity = profanityCheck(profaneWordsIdentified = result,
    #                                         profaneWordsthreshold = str(self.Profanity_threshold),
    #                                         result = 'PASSED')
    #             self.dict_profanity['status'] = True
            
    #         else:
    #             obj_profanity = profanityCheck(profaneWordsIdentified = result,
    #                                         profaneWordsthreshold = str(self.Profanity_threshold),
    #                                         result = 'FAILED')
    #             self.dict_profanity['status'] = False

    #         self.dict_profanity['object'] = obj_profanity
    #         et = time.time()
    #         rt = et - st
    #         dictcheck["Profanity Check"]=str(round(rt,3))+"s"
    #         log.debug(f"profanity run time: {rt}")
    #         self.timecheck["Profanity Check"]=str(round(rt,3))+"s"

    #         return self.dict_profanity
    #     except Exception as e:
    #         log.error("Failed at validate profanity")
    #         log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
    #                                                "Error Module":"Failed at validate profanity"})
    #         # log.error(f"Exception: {e}")
    #         log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

    def toxicity_val(self,result,rounded_toxic,list_toxic,st,checkRes):
                #print("---------------------ToxicityVal----------------------")
                if result < self.ToxicityThreshold:
                    obj_toxicity = toxicityCheck(toxicityScore =rounded_toxic,
                                                toxicitythreshold = str(self.ToxicityThreshold),
                                                result = 'PASSED')
                    self.dict_toxicity['status'] = True

                else:
                    obj_toxicity = toxicityCheck(toxicityScore = list_toxic,
                                                toxicitythreshold = str(self.ToxicityThreshold),
                                                result = 'FAILED')
                    self.dict_toxicity['status'] = False

                self.dict_toxicity['object'] = obj_toxicity
            
                et = time.time()
                rt = et - st
                dictcheck["Toxicity Check"]=str(round(rt,3))+"s"
                log.info(f"toxicity run time: {rt}")
                self.timecheck["Toxicity Check"]=str(round(rt,3))+"s"
                checkRes.append(self.dict_toxicity)

    def profanity_val(self,result,st,checkRes):
                    #print("---------------------ProfanityVal----------------------")
                    profRes=[]
                    if result > 0.6:
                        res = profanity.censor(self.text)
                        # #print("==",res)
                        
                        profRes=res[1]
                    # self.modelcall["Profanity Check"]=mt
                    self.dict_profanity['key'] = 'Profanity Check'
                    if len(profRes) < self.Profanity_threshold:
                        obj_profanity = profanityCheck(profaneWordsIdentified = profRes,
                                                    profaneWordsthreshold = str(self.Profanity_threshold),
                                                    result = 'PASSED')
                        self.dict_profanity['status'] = True

                    else:
                        obj_profanity = profanityCheck(profaneWordsIdentified = profRes,
                                                    profaneWordsthreshold = str(self.Profanity_threshold),
                                                    result = 'FAILED')
                        self.dict_profanity['status'] = False

                    self.dict_profanity['object'] = obj_profanity
                    et = time.time()
                    rt = et - st
                    dictcheck["Profanity Check"]=str(round(rt,3))+"s"
                    log.debug(f"profanity run time: {rt}")
                    self.timecheck["Profanity Check"]=str(round(rt,3))+"s"
                    checkRes.append(self.dict_profanity)
 
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
            
            # self.modelcall["Toxicity Check"]=mt
            self.dict_toxicity['key'] = 'Toxicity Check'
            self.modeltime['Toxicity Check']=modelcalltime
            list_toxic = []
            list_toxic.append(toxic_dict)
            rounded_toxic = []
            for item in list_toxic:
                toxic_score = item['toxicScore']
                rounded_score = [{'metricName': score['metricName'], 'metricScore': round(score['metricScore'], 3)} for score in toxic_score]
                rounded_item = {'toxicScore': rounded_score}
                rounded_toxic.append(rounded_item)
            checkRes=[]
            threads=[]
            if("Toxicity" in self.Checks_selected):
                #print("---------------------Tocixity----------------------")
                thread=threading.Thread(target=self.toxicity_val,args=(result,rounded_toxic,list_toxic,st,checkRes))
                thread.start()
                threads.append(thread)
            if("Profanity" in self.Checks_selected):
                    #print("---------------------Profanity----------------------")
                    thread=threading.Thread(target=self.profanity_val,args=(result,st,checkRes))
                    thread.start()
                    threads.append(thread)
            # for thread in threads:
            #     thread.start()
            for thread in threads:
                thread.join()
                    
            return checkRes
        except Exception as e:
            log.error("Failed at validate toxicity")
           
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at validate toxicity"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
    

    
    # async def validate_refusal(self,headers):
    #     try:
    #         log.info(f"Initialising Refusal validation")
    #         st = time.time()
    #         refusal = Refusal()
    #         result = await refusal.refusal_check(self.text,headers)
    #         self.dict_refusal['key'] = 'Refusal Check'
    #         if result <= self.RefusalThreshold:
    #             obj_refusal= refusalCheck(refusalSimilarityScore = str(round(float(result),2)),
    #                                         RefusalThreshold = str(self.RefusalThreshold),
    #                                         result = 'PASSED')
    #             self.dict_refusal['status'] = True
    #         else:
    #             obj_refusal = refusalCheck(refusalSimilarityScore =  str(round(float(result),2)),
    #                                         RefusalThreshold = str(self.RefusalThreshold),
    #                                         result = 'FAILED')
    #             self.dict_refusal['status'] = False

    #         self.dict_refusal['object'] = obj_refusal
    #         et = time.time()
    #         rt = et - st
    #         dictcheck["Refusal Check"]=str(round(rt,3))+"s"
    #         log.debug(f"refusal run time: {rt}")
    #         self.timecheck["Refusal Check"]=str(round(rt,3))+"s"
    #         return self.dict_refusal
        
    #     except Exception as e:
    #         log.error("Failed at validate refusal")
    #         log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
    #                                                "Error Module":"Failed at validate refusal"})
    #         # log.error(f"Exception: {e}")
    #         log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

    async def validate_text_relevance(self,output_text,headers):
        try:
            log.info(f"Initialising Text relevance validation")
            st = time.time()
            self.dict_relevance['key']="Text Relevance Check"
            prSimilarity = promptResponse()
            prSimilarityscore = await prSimilarity.promptResponseSimilarity(output_text,self.text,headers)
            self.dict_relevance['status']=True
            self.dict_relevance['object']=textRelevanceCheck(PromptResponseSimilarityScore = str(int(prSimilarityscore*100)))
            rt = time.time()-st
            dictcheck["Text Relevance Check"]=str(round(rt,3))+"s"
            log.debug(f"Text relevance run time: {rt}")
            self.timecheck["Text Relevance Check"]=str(round(rt,3))+"s"

            return [self.dict_relevance]
        except Exception as e:
            log.error("Failed at validate_text_relevance")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at validate_text_relevance"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

    async def validate_text_quality(self):
        try:
            log.info(f"Initialising Text quality validation")
            st = time.time()
            self.dict_textQuality['key']="Text Quality Check"
            readabilityScore,textGrade = text_quality(self.text)
            
            self.dict_textQuality['status']=True
            self.dict_textQuality['object']=textQuality(readabilityScore = str(int(readabilityScore)),
                                                        textGrade=str(textGrade))
            et = time.time()
            rt = et - st
            dictcheck["Text Quality Check"]=str(round(rt,3))+"s"
            log.debug(f"Text quality run time: {rt}")
            self.timecheck["Text Quality Check"]=str(round(rt,3))+"s"
            return [self.dict_textQuality]
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
                # 'JailBreak':"self.validate_jailbreak(headers)",
                # 'Toxicity':"self.validate_toxicity(headers)",
                'Piidetct':"self.validate_pii(headers)",
                # 'Profanity':"self.validate_profanity(headers)",
                # "CustomizedTheme":"self.validate_customtheme(theme,headers)",
                # 'RestrictTopic':"self.validate_restrict_topic(self.config_details,headers)",
                # 'Refusal' : "self.validate_refusal(headers)",
                'TextRelevance' : "self.validate_text_relevance(output_text,headers)",
                'TextQuality' : "self.validate_text_quality()",
                'randomNoiseCheck':'self.validate_smoothllm(headers)',
                'advancedJailbreakCheck':'self.validate_bergeron(headers)',
                'Sentiment':"self.validate_sentiment(headers)",
                'InvisibleText':"self.validate_invisibletext(headers)",
                'Gibberish':"self.validate_gibberish(headers)",
                'BanCode':"self.validate_bancode(headers)",
                }
            profanFlag=1
            jailFlag=1
            
            for i in self.Checks_selected:
                    if(i == "Profanity" or i=="Toxicity"):
                        if(profanFlag==1):
                            profanFlag=0
                            tasks.append(self.validate_toxicity(headers))
                    elif(i == "JailBreak" or i=="Refusal" or i=="CustomizedTheme"):
                        if(jailFlag==1):
                            jailFlag=0
                            tasks.append(self.validate_customtheme(theme,headers))
                    elif("RestrictTopic" in i):
                        if("RestrictTopic-lite" in i):
                            tasks.append(self.validate_restrict_topic(self.config_details,headers,model="nliMini"))
                        else:    
                            tasks.append(self.validate_restrict_topic(self.config_details,headers,model="dberta"))
                            
                    else:
                        tasks.append(eval(checkdict[i]))
            for i in llm_BasedChecks:
                    tasks.append(eval(checkdict[i]))

            results = await asyncio.gather(*tasks)
            
            list_tasks = []
            results=sum(results, [])
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



def callModerationModels(text,payload,headers,deployment_name=None,output_text=None,llm_BasedChecks=[]):
    global startupFlag,jailbreak_embeddings,refusal_embeddings,topic_embeddings
    list_checks = []
    payload=AttributeDict(payload)
    portfolio = payload.PortfolioName if "PortfolioName" in payload else "None"
    accountname = payload.AccountName if "AccountName" in payload else "None"
    payload.ModerationCheckThresholds=AttributeDict(payload.ModerationCheckThresholds)
    theme=AttributeDict(payload.ModerationCheckThresholds.CustomTheme)
    emoji_mod_opt=payload.EmojiModeration if "EmojiModeration" in payload else "no" #for emoji moderation

    validate_input=validation_input(deployment_name,text,payload,emoji_mod_opt,accountname,portfolio)
    passed_text,dict_all=asyncio.run(validate_input.main(theme,output_text,headers,llm_BasedChecks))
    
    check_obs = {'Prompt Injection Check': validate_input.dict_prompt['object'],
                 'Jailbreak Check':validate_input.dict_jailbreak['object'],
                 'Profanity Check':validate_input.dict_profanity['object'],
                 'Privacy Check':validate_input.dict_privacy['object'],
                 'Toxicity Check':validate_input.dict_toxicity['object'],
                 'Restricted Topic Check':validate_input.dict_topic['object'],
                 'Custom Theme Check':validate_input.dict_customtheme['object'],
                 'Text Quality Check':validate_input.dict_textQuality['object'],
                 'Refusal Check':validate_input.dict_refusal['object'],
                 'Text Relevance Check':validate_input.dict_relevance['object'],
                 'Random Noise Check':validate_input.dict_smoothllm['object'],
                 'Advanced Jailbreak Check':validate_input.dict_bergeron['object'],
                 'Sentiment Check':validate_input.dict_sentiment['object'],
                 'Invisible Text Check':validate_input.dict_invisibleText['object'],
                 'Gibberish Check':validate_input.dict_gibberish['object'],
                 'Ban Code Check':validate_input.dict_bancode['object'],
                 'model time':validate_input.modeltime,
                 'time check':validate_input.timecheck}
    
    status = 'PASSED'
    for i in dict_all:
        if i['status']==False:
            status = 'FAILED'
            list_checks.append(i['key'])

    objSummary = summary(status = status,reason = list_checks)
    log.debug(f'objSummary:{objSummary}')
    check_obs['summary'] = objSummary
    return check_obs


#===================================  For Decoupled Moderation  ============================================#
class moderation:
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def completions(payload,headers,deployment_name=None,output_text=None,llm_BasedChecks=[],translate=None) -> dict:
        try:    
            
            lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"
            created = datetime.now()
            text = payload.Prompt

            if translate == "google" or translate == "yes":
                print("Inside Google Translate")
                starttime = time.time()
                text,lang = translator.translate(payload.Prompt)
                endtime = time.time()
                rt = endtime - starttime
                dict_timecheck["translate"]=str(round(rt,3))+"s"
            elif translate == "azure":
                print("Inside Azure Translate")
                starttime = time.time()
                text,lang = translator.translate(payload.Prompt)
                endtime = time.time()
                rt = endtime - starttime
                dict_timecheck["translate"]=str(round(rt,3))+"s"
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
     
            obj = callModerationModels(text,payload,headers,deployment_name,output_text,llm_BasedChecks)
      
            obj_requestmoderation = RequestModeration(text = text,
                                                        promptInjectionCheck = obj['Prompt Injection Check'],
                                                        jailbreakCheck= obj['Jailbreak Check'],
                                                        privacyCheck = obj['Privacy Check'],
                                                        profanityCheck = obj['Profanity Check'],
                                                        toxicityCheck = obj['Toxicity Check'],
                                                        restrictedtopic = obj['Restricted Topic Check'],
                                                        customThemeCheck = obj['Custom Theme Check'],
                                                        textQuality =obj['Text Quality Check'],
                                                        refusalCheck = obj['Refusal Check'],
                                                        sentimentCheck=obj['Sentiment Check'],
                                                        invisibleTextCheck = obj['Invisible Text Check'],
                                                        gibberishCheck = obj['Gibberish Check'],
                                                        bancodeCheck=obj['Ban Code Check'],
                                                        summary = obj['summary'])
            
            obj_ModerationResults = ModerationResults(lotNumber=lotNumber,created=str(created) ,moderationResults = obj_requestmoderation)
        
            # log.info("res="+str(obj_ModerationResults)+str(obj['time check'])+str(obj['model time']))
            return obj_ModerationResults,obj['time check'],obj['model time']
        except Exception as e:
            print(e)
            log.error("Failed at Completion call Function")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Completion Function"})
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

            



#========================================= For Coupled Moderation  =========================================# 
class coupledModeration:
    
    @lru.lru_cache(ttl=cache_ttl,size=cache_size,flag=cache_flag)
    def coupledCompletions(payload,token):
        smoothllmresponse = smoothLlmCheck(smoothLlmScore="",smoothLlmThreshold = "",result = 'UNMODERATED')
        bergeronResponse = bergeronCheck(text="",result = 'UNMODERATED')
        objprofanity_out = profanityCheck(profaneWordsIdentified=[],profaneWordsthreshold = '0',result = 'UNMODERATED')
        objprivacy_out = privacyCheck(entitiesRecognised=[],entitiesConfiguredToBlock = [],result = 'UNMODERATED')
        # objtoxicity_out = toxicityCheck(toxicityScore= [],toxicitythreshold = '',result = 'UNMODERATED')
        # objtopic_out = restrictedtopic(topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtoxicity_out = toxicityCheckTypes(toxicityTypesRecognised = [],
									toxicityTypesConfiguredToBlock=[t.value for t in TOXICITYTYPES][0:-1],
									toxicityScore= [],
									toxicitythreshold = '0',
									result = 'UNMODERATED')		
        topicTypesConfiguredToBlock = payload.ModerationCheckThresholds['RestrictedtopicDetails']['Restrictedtopics']
        objtopic_out = restrictedtopicTypes(topicTypesConfiguredToBlock=topicTypesConfiguredToBlock,
                                                    topicTypesRecognised=[],
                                                    topicScores=[],topicThreshold="0",result = "UNMODERATED")
        objtextQuality_out = textQuality(readabilityScore = "0",textGrade="")
        objpromptResponse_out = textRelevanceCheck(PromptResponseSimilarityScore = "0")
        objrefusal_out = refusalCheck(refusalSimilarityScore = "" , RefusalThreshold = "" , result = 'UNMODERATED')
        obj_sentiment_out = sentimentCheck(score = "",threshold = "",result = 'UNMODERATED')
        obj_invisibleText_out = invisibleTextCheck(invisibleTextIdentified=[],threshold = "",result = 'UNMODERATED')
        obj_gibberish_out = gibberishCheck(gibberishScore=[],threshold = "",result = 'UNMODERATED')
        obj_bancode_out = bancodeCheck(score=[],threshold = "",result = 'UNMODERATED')
        list_choices = []
        created = datetime.now()
        global dictcheck
        st = time.time()
            
        llm_Based_Checks = payload.llm_BasedChecks
        emojiModOpt=payload.EmojiModeration if "EmojiModeration" in payload else "no"
        deployment_name = payload.model_name if "model_name" in payload else "gpt4"
        translate = payload.translate if "translate" in payload else None
        text = payload.Prompt
        PromptTemplate=payload.PromptTemplate
        temperature = float(payload.temperature)
        LLMinteraction = payload.LLMinteraction
        userid = payload.userid if "userid" in payload else "None"
        lotNumber = str(payload.lotNumber) if "lotNumber" in payload else "None"

        if translate == "google" or translate == "yes":
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
                rt = endtime -