'''
Copyright 2024 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import traceback
import json
import typing as t
from config.prompt_templates import *
import openai
import time
import datetime
from config.logger import CustomLogger
from openai import AzureOpenAI
import os
log = CustomLogger()
prompt_template = {}

# RA-453 and RA-312 : Class created for evaluation by LLM (Template-based Approach)
class CustomEvaluation():
    detection_type: str = ""
    userId: str = ""
    deployment_name: str = ""
    temperature: str = ""
    col_question: str = "question"
    col_response: str = "response"
    col_out: str = "score"
    scenario_description: t.Optional[str] = None

    def __init__(self, detection_type,userId,deployment_name,temperature):
        self.detection_type=detection_type
        self.userId=userId
        self.deployment_name=deployment_name
        self.temperature=temperature
        
    def run(self,data):
        #data_send = polars_to_json_serializable_dict(data)
        data_send = data
        for row in data_send:
            row["question"] = row.pop(self.col_question)
            if "response" in list(data_send[0].keys()):
                row["response"] = row.pop(self.col_response)
            
        retries = 0
        max_retries = 10
        while retries < max_retries:
            try:
                results = self.evaluate_customized(data_send)
                assert results is not None
                # return {
                #     "output": pl.from_dicts(results)
                # }
                return results
            except openai.RateLimitError as e:
                    retries += 1
                    if(retries > max_retries):
                        return "Rate Limit Error"
                    wait_time = 2 ** retries  # Exponential backoff
                    print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)      
  
            except openai.BadRequestError as IR:
                log.error(f"Exception: {IR}")
                return str(IR)
            
            except Exception as e:
                log.error("Error occurred in Evaluation by LLM")
                return str(e)
   
    
    def validate_func_for_evaluation(self, llm_output):
        is_correct = True
        is_correct = is_correct and ("score" in llm_output)
        is_correct = is_correct and ("analysis" in llm_output)
        return is_correct
    

    def get_templates(self, detection_type:str,userId:str,question:str,response:str):
        try:
            template = {"detection_criteria": DETECTION_CRITERIA.replace("{detection_type}", self.detection_type),
                        "detection_type":detection_type,
                        "output_format":OUTPUT_FORMAT,
                        "task_data":TASK_DATA_FOR_REQ.replace("{question}",question)}
            
            if detection_type == "NAVI_TONEMODERATION_CORRECTNESS":
                template.update({"detection_criteria":"",
                                 "evaluation_criteria":EVALUATION_CRITERIA_NAVI_TONEMODERATION_CORRECTNESS,
                                "prompting_instructions":PROMPTING_INSTRUCTIONS_NAVI_TONEMODERATION_CORRECTNESS,
                                "few_shot_examples":FEW_SHOT_NAVI_TONEMODERATION_CORRECTNESS })
                
            elif detection_type == "PROMPT_INJECTION":
                template.update({"evaluation_criteria":EVALUATION_CRITERIA_PROMPT_INJECTION,
                                "prompting_instructions":PROMPTING_INSTRUCTIONS_PROMPT_INJECTION,
                                "few_shot_examples":FEW_SHOT_PROMPT_INJECTION })
                
            elif detection_type == "JAILBREAK":
                template.update({ "evaluation_criteria":EVALUATION_CRITERIA_JAILBREAK,
                                "prompting_instructions":PROMPTING_INSTRUCTIONS_JAILBREAK,
                                "few_shot_examples":FEW_SHOT_JAILBREAK })

            elif detection_type == "FAIRNESS_AND_BIAS":
                template.update({"evaluation_criteria":EVALUATION_CRITERIA_FAIRNESS,
                                "prompting_instructions":PROMPTING_INSTRUCTIONS_FAIRNESS,
                                "few_shot_examples":FEW_SHOT_FAIRNESS })
                
            elif detection_type == "LANGUAGE_CRITIQUE_COHERENCE":
                template.update({"evaluation_criteria":EVALUATION_CRITERIA_LANGUAGE_CRITIQUE_COHERENCE,
                                "prompting_instructions":PROMPTING_INSTRUCTIONS_LANGUAGE_CRITIQUE_COHERENCE,
                                "few_shot_examples":FEW_SHOT_LANGUAGE_CRITIQUE_COHERENCE })
            
            elif detection_type == "LANGUAGE_CRITIQUE_FLUENCY":
                template.update({"evaluation_criteria":EVALUATION_CRITERIA_LANGUAGE_CRITIQUE_FLUENCY,
                                "prompting_instructions":PROMPTING_INSTRUCTIONS_LANGUAGE_CRITIQUE_FLUENCY,
                                "few_shot_examples":FEW_SHOT_LANGUAGE_CRITIQUE_FLUENCY })  
                
            elif detection_type == "LANGUAGE_CRITIQUE_GRAMMAR":
                template.update({"evaluation_criteria":EVALUATION_CRITERIA_LANGUAGE_CRITIQUE_GRAMMAR,
                                "prompting_instructions":PROMPTING_INSTRUCTIONS_LANGUAGE_CRITIQUE_GRAMMAR,
                                "few_shot_examples":FEW_SHOT_LANGUAGE_CRITIQUE_GRAMMAR })   
                
            elif detection_type == "LANGUAGE_CRITIQUE_POLITENESS":
                template.update({"evaluation_criteria":EVALUATION_CRITERIA_LANGUAGE_CRITIQUE_POLITENESS,
                                "prompting_instructions":PROMPTING_INSTRUCTIONS_LANGUAGE_CRITIQUE_POLITENESS,
                                "few_shot_examples":FEW_SHOT_LANGUAGE_CRITIQUE_POLITENESS }) 

            
            # Response Related Checks
            elif detection_type == "RESPONSE_COMPLETENESS":
                tasks = {"question":question,"response":response}
                template.update({"evaluation_criteria":EVALUATION_CRITERIA_RESPONSE_COMPLETENESS,
                                    "prompting_instructions":PROMPTING_INSTRUCTIONS_RESPONSE_COMPLETENESS,
                                    "few_shot_examples":FEW_SHOT_RESPONSE_COMPLETENESS,
                                    "task_data":TASK_DATA_FOR_RESP.format(**tasks)}) 
                
            elif detection_type == "RESPONSE_CONCISENESS":
                tasks = {"question":question,"response":response}
                template.update({"evaluation_criteria":EVALUATION_CRITERIA_RESPONSE_CONCISENESS,
                                    "prompting_instructions":PROMPTING_INSTRUCTIONS_RESPONSE_CONCISENESS,
                                    "few_shot_examples":FEW_SHOT_RESPONSE_CONCISENESS,
                                    "task_data":TASK_DATA_FOR_RESP.format(**tasks)})
                    
                
            elif detection_type == "RESPONSE_LANGUAGE_CRITIQUE_COHERENCE":
                tasks = {"question":question,"response":response}
                template.update({"evaluation_criteria":EVALUATION_CRITERIA_LANGUAGE_CRITIQUE_COHERENCE,
                                    "prompting_instructions":PROMPTING_INSTRUCTIONS_LANGUAGE_CRITIQUE_COHERENCE,
                                    "few_shot_examples":FEW_SHOT_LANGUAGE_CRITIQUE_COHERENCE,
                                    "task_data":TASK_DATA_FOR_RESP.format(**tasks)})
            
            elif detection_type == "RESPONSE_LANGUAGE_CRITIQUE_FLUENCY":
                tasks = {"question":question,"response":response}
                template.update({"evaluation_criteria":EVALUATION_CRITERIA_LANGUAGE_CRITIQUE_FLUENCY,
                                    "prompting_instructions":PROMPTING_INSTRUCTIONS_LANGUAGE_CRITIQUE_FLUENCY,
                                    "few_shot_examples":FEW_SHOT_LANGUAGE_CRITIQUE_FLUENCY,
                                    "task_data":TASK_DATA_FOR_RESP.format(**tasks)})  
                    
            elif detection_type == "RESPONSE_LANGUAGE_CRITIQUE_GRAMMAR":
                tasks = {"question":question,"response":response}
                template.update({"evaluation_criteria":EVALUATION_CRITERIA_LANGUAGE_CRITIQUE_GRAMMAR,
                                    "prompting_instructions":PROMPTING_INSTRUCTIONS_LANGUAGE_CRITIQUE_GRAMMAR,
                                    "few_shot_examples":FEW_SHOT_LANGUAGE_CRITIQUE_GRAMMAR,
                                    "task_data":TASK_DATA_FOR_RESP.format(**tasks)})   
                    
            elif detection_type == "RESPONSE_LANGUAGE_CRITIQUE_POLITENESS":
                tasks = {"question":question,"response":response}
                template.update({"evaluation_criteria":EVALUATION_CRITERIA_LANGUAGE_CRITIQUE_POLITENESS,
                                    "prompting_instructions":PROMPTING_INSTRUCTIONS_LANGUAGE_CRITIQUE_POLITENESS,
                                    "few_shot_examples":FEW_SHOT_LANGUAGE_CRITIQUE_POLITENESS,
                                    "task_data":TASK_DATA_FOR_RESP.format(**tasks)})
            

            else:
                    log.info("inside else block")
                    found = False
                    data = prompt_template[userId]
                    for d in data:
                        if d['templateName'] == detection_type:
                            found=True
                            for s in d['subTemplates']:
                                template[s['template']] = s['templateData']
                            break
                    
                    if not found:
                        log.error(f"Invalid Detection type : {detection_type}")
                        raise Exception("Invalid Detection Type")

            return template 
        
        except Exception as e:
            log.error(f"Invalid Detection type : {detection_type}")
            return str(e)
            
    

    # def get_templates(self, detection_type:str):
    #     template = {"detection_criteria": DETECTION_CRITERIA.replace("{detection_type}", self.detection_type),
    #                     "detection_type":detection_type,
    #                     "output_format":OUTPUT_FORMAT}
    #     with open(r"./templates/data/templates.json","r") as f:
    #         data = json.load(f)
        
    #         for d in data['templates']:
    #             if d['templateName'] == detection_type:
    #                 for s in d['subTemplates']:
    #                     template[s['template']] = s['templateData']
    #                 break
        
    #     return template 
        
    
    
    def evaluate_customized(self,data):
        st = datetime.datetime.now()
        #input_payloads = []
        validation_func = self.validate_func_for_evaluation
        output_payloads = []
        for idx, row in enumerate(data):
            kwargs = row
            
            row["response"] = row.pop(self.col_response) if "response" in list(data[0].keys()) else ""
            kwargs.update(self.get_templates(self.detection_type,
                                             self.userId,
                                             row["question"],
                                             row["response"]))
           
            
            # ------------------ Added Azure Open AI Client ----------------------------- #
            if self.deployment_name == "gpt3":
                deployment_name = os.getenv("OPENAI_MODEL_GPT3")
                openai_api_base = os.getenv("OPENAI_API_BASE_GPT3")
                openai_api_key = os.getenv("OPENAI_API_KEY_GPT3")
                openai_api_version = os.getenv("OPENAI_API_VERSION_GPT3")
            else:
                deployment_name = os.getenv("OPENAI_MODEL_GPT4")
                openai_api_base = os.getenv("OPENAI_API_BASE_GPT4")
                openai_api_key = os.getenv("OPENAI_API_KEY_GPT4")
                openai_api_version = os.getenv("OPENAI_API_VERSION_GPT4")
            client = AzureOpenAI(api_key=openai_api_key, 
                                 azure_endpoint=openai_api_base,
                                 api_version=openai_api_version)
            messages = [{"role": "user", "content": GENERAL_PROMPT_TEMPLATE.format(**kwargs)}]

            response = client.chat.completions.create(
                model=deployment_name,
                messages = messages,
                temperature=self.temperature,
                max_tokens=500)
            
            output_payloads.append(response)
            

        results = []
        try:
            for idx,res in enumerate(output_payloads):
                if validation_func(res.choices[0].message.content) != True:
                    log.info("Failed Validation.")

                #idx = res.metadata["index"]
                output = {
                    "score": None,
                    "threshold":None,
                    "analysis": None,
                    "bias_type": None,
                    "group": None,
                    "result":None,
                    "Tone Score":None,
                    "role":None,
                    "Sentiment":None,
                    "Domain":None,
                    "Context":None,
                    "outputBeforemoderation":None,
                    "outputAfterSentimentmoderation":None
                }

                output["threshold"] = 60
                
                
                if res.choices[0].finish_reason=="content_filter":
                    output["score"] = 100
                    output["analysis"] = "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766"
                else:
                    output["score"] = int(json.loads(res.choices[0].message.content)["score"])
                    output["analysis"] = json.loads(res.choices[0].message.content)["analysis"]

                if self.detection_type == "NAVI_TONEMODERATION_CORRECTNESS":
                    output["Tone Score"] = json.loads(res.choices[0].message.content)["Tone Score"]
                    output["role"] = json.loads(res.choices[0].message.content)["role"]
                    output["Sentiment"] = json.loads(res.choices[0].message.content)["Sentiment"]
                    output["Context"] = json.loads(res.choices[0].message.content)["Context"]
                    output["Domain"] = json.loads(res.choices[0].message.content)["Domain"]
                    output["outputBeforemoderation"] = json.loads(res.choices[0].message.content)["outputBeforemoderation"]
                    output["outputAfterSentimentmoderation"] = json.loads(res.choices[0].message.content)["outputAfterSentimentmoderation"]
                    output["analysis"] = json.loads(res.choices[0].message.content)["outputAfterSentimentmoderation"]
                else:
                    del output["Tone Score"]
                    del output["role"]
                    del output["Sentiment"]
                    del output["Domain"]
                    del output["outputBeforemoderation"]
                    del output["outputAfterSentimentmoderation"]
                    del output["Context"]

                if self.detection_type == "FAIRNESS_AND_BIAS":
                    output["bias_type"] = json.loads(res.choices[0].message.content)["bias_type"]
                    output["group"] = json.loads(res.choices[0].message.content)["group"]
                else:
                    del output["bias_type"]
                    del output["group"]
                    
                    
                if output["score"] > output["threshold"]:
                    output["result"] = "FAILED"
                else:
                    output["result"] = "PASSED"
                  
                output["timetaken"]=str(round((datetime.datetime.now() - st).total_seconds(),3))+"s"
                results.append((idx, output))
            
            results = [val for _, val in sorted(results, key=lambda x: x[0])]

            return results    
        
        except openai.BadRequestError as IR:
                log.error(f"Exception: {IR}")
                log.error("Invalid Request Error")
                return str(IR)
            
        except Exception as e:
                line_number = traceback.extract_tb(e.__traceback__)[0].lineno
                log.error(f"Exception: {line_number,e}")
                return str(e)
