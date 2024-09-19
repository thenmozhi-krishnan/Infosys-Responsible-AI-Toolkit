'''
Copyright 2024 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
########################################### IMPORT LIBRARIES ############################################

from langchain.prompts import PromptTemplate
import openai
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from config.prompt_templates import *
from config.logger import CustomLogger, request_id_var
import os
import json
from utilities.utility_methods import *
from utilities.lruCaching import *
import traceback
import threading
from telemetry import telemetry
if os.getenv("DBTYPE") != "False":
    from dao.AdminDb import Results


########################################  Prompt Template  ##########################################
BASELINE_PROMPT = """You are a detail-oriented and highly analytical LLM to detect {detection_type} in the provided prompt and image(if provided).
        {evaluation_criteria}
        {prompting_instructions}
        {few_shot}
        {output_format}
        {task_data} 
        """


NORMAL_OUTPUT = """
Given the below User Query , generate an output with following fields:
"analysis": "[keep it crisp and to the point, including all necessary details]",
"score": [Assign a decimal score between 0-1]
"""

FAIRNESS_OUTPUT = """
Given the below User Query , generate an output with following fields:
"analysis": "[keep it crisp and to the point, including all necessary details]",
"score": [Assign a decimal score between 0-1],
"bias_type": "[mentions the bias type(s).The value here should be comma separated clear bias type(s), state None in case of no clear bias type(s)]",
"group": "[mentions the group(s) towards which there is a bias.The value here should be comma separated clear group(s), state None in case of no clear group(s)]"
"""

NAVI_OUTPUT = """
Given the below User Query , generate an output with following fields:
"analysis": "[keep it crisp and to the point, including all necessary details]",
"score": [Assign a decimal score between 0-1],
"role":"[role of LLM eg: expert,Advisory,Support]",
"Sentiment":"[this should show emotion state of user prompt eg: Positive - Upbeat]",
"Context":"[this will be coming context]"
"Domain":"[This should show Domain of topic]",
"outputBeforemoderation":"[This should show output before moderation]",
"outputAfterSentimentmoderation":"[This should show output before moderation]"
"""

TOXICITY_OUTPUT= """
Given the below User Query , generate an output with following fields:
"analysis": "[keep it crisp and to the point, including all necessary details]",
"score": [
           {
            "metricName" : "Toxicity",
            "metricScore" :[Assign a decimal score between 0-1]
           },
           {
            "metricName" : "Severe Toxicity",
            "metricScore" :[Assign a decimal score between 0-1]
           },
           {
            "metricName" : "Insult",
            "metricScore" :[Assign a decimal score between 0-1]
           },
           {
            "metricName" : "Threat",
            "metricScore" :[Assign a decimal score between 0-1]
           },
           {
            "metricName" : "Obscene",
            "metricScore" :[Assign a decimal score between 0-1]
           },
           {
            "metricName" : "Hate speech",
            "metricScore" :[Assign a decimal score between 0-1]
           },
           {
            "metricName" : "Identity Attack",
            "metricScore" :[Assign a decimal score between 0-1]
           },
           {
            "metricName" : "Sexual Explicit",
            "metricScore" :[Assign a decimal score between 0-1]
           }
],
"category": "[mention the category of toxicity]"
"""

RESTRICTED_TOPIC_OUTPUT = """
Given the below User Query , generate an output with following fields:
"analysis": "[give detailed explanation about the image and user query.Also explain what is Restricted Topic and whether the user query and image is having any restricted topic or not.]",
"score": [Assign a decimal score between 0-1]
"category": "[mention the restricted topic being used]"
"""

TASK_DATA_FOR_REQ = """
Task Data.
[User Query]: {question}
[Output]:
"""

TASK_DATA_FOR_RESP = """
Task Data.
[User Query]: {question}
[Response]: {response}
[Output]:
"""
#######################################################################################################


log = CustomLogger()
template_log_dict={}
cache_ttl = int(os.getenv("CACHE_TTL"))
cache_size = int(os.getenv("CACHE_SIZE"))
# evalLLMtelemetryurl=os.getenv("EVALLLMTELEMETRYPATH")


###############################   Template based Guardrails (LLM Evaluation) #######################


@lru_cache(ttl=cache_ttl,size=cache_size)
def get_gpt_response(prompt,modelName):
    MODEL_NAME,API_BASE,API_KEY,API_VERSION,API_TYPE = config(modelName)
    client = AzureOpenAI(api_key=API_KEY, 
                         azure_endpoint=API_BASE,
                         api_version=API_VERSION)
    resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages = [{"role": "user", "content": prompt}] ,
                temperature=0,
                max_tokens=500)
    llm_resp = resp.choices[0].message.content if len(resp.choices[0].message.content)!=0 else resp.choices[0].finish_reason
    return llm_resp



def get_output_format(template_name):
    if template_name == "Toxicity Check":
        return TOXICITY_OUTPUT
    elif template_name == "Restricted Topic Check":
        return RESTRICTED_TOPIC_OUTPUT
    elif template_name == "Fairness and Bias Check":
        return FAIRNESS_OUTPUT
    elif template_name == "Navi Tone Correctness Check":
        return NAVI_OUTPUT
    else:
        return NORMAL_OUTPUT
    


@lru_cache(ttl=cache_ttl,size=cache_size)
def get_response(text,template_name,userId,modelName):
    MODEL_NAME,API_BASE,API_KEY,API_VERSION,API_TYPE = config(modelName)
    try:
        llm_resp = ""
        llm = AzureChatOpenAI(deployment_name=MODEL_NAME,
                                    openai_api_version=API_VERSION,
                                    openai_api_key=API_KEY,
                                    azure_endpoint=API_BASE,
                                    openai_api_type = API_TYPE,
                                    temperature = 0)
        
        final_prompt = PromptTemplate.from_template(BASELINE_PROMPT)
        final_chain = final_prompt | llm
        vars = get_templates(template_name,userId)

        if template_name.startswith("Response"):
             llm_resp = get_gpt_response(text,MODEL_NAME)


        response = final_chain.invoke({f"detection_type":template_name,
                                    "evaluation_criteria":vars["evaluation_criteria"],
                                    "prompting_instructions":vars["prompting_instructions"],
                                    "few_shot":vars["few_shot_examples"],
                                    "output_format":get_output_format(template_name),
                                    "task_data":TASK_DATA_FOR_RESP.format(**{"question":text,"response":llm_resp}) if template_name.startswith("Response") else TASK_DATA_FOR_REQ.replace("{question}",text)})
        
        content = response.dict()['content']
        for pattern in ("[Output]:\n","[Output] :\n"):
            content = content.replace(pattern,"")
        content = content.replace("{{","{").replace("}}","}")   

        if(text==""):
            log.info("Prompt is Empty")
            template_log_dict[request_id_var.get()].append("Prompt is Empty")
            return "Error Occured due to empty prompt" 
        
        print("content : ",content)
        response_dict = json.loads(content)
        
        response_dict['threshold']=0.6
        response_dict['score']=float(response_dict['score'])
        response_dict['result']="FAILED" if response_dict['score'] > response_dict['threshold'] else "PASSED"
       
        if response.dict()['response_metadata']['finish_reason'] == "content_filter":
            response_dict['analysis'] = "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766"
            response_dict['score'] = -1
            response_dict['result'] = "Can't be determined"
        
        return response_dict
                
    except Exception as e:
        line_number = traceback.extract_tb(e.__traceback__)[0].lineno
        template_log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at Eval LLM model call"})
        log.error(f"Error occured in Chain to generate initial answer: {line_number,e}")




class TextTemplateService:
    def generate_response(self,id,req):
        try:
            template_log_dict[request_id_var.get()]=[]
            AccountName=req.AccountName if "AccountName" in req else "None"
            PortfolioName=req.PortfolioName if "PortfolioName" in req else "None"
            userid = req.userid if "userid" in req else "None"
            return get_response(req['Prompt'],req['template_name'],userid,req['model_name'])

        except Exception as e:
            line_number = traceback.extract_tb(e.__traceback__)[0].lineno
            template_log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                    "Error Module":"Failed at Eval LLM model call"})
            log.error(f"Error occured in Chain to generate initial answer: {line_number,e}")


