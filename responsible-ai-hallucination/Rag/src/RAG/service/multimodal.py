import base64
from io import BytesIO
from PIL import Image
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage,ChatMessage
from RAG.service.service import cache,MAX_CACHE_SIZE,fs, dbtypename
import os
import time
import traceback
import requests
import json
from RAG.config.logger import CustomLogger,request_id_var

log=CustomLogger()
request_id_var.set("Startup")

try:
    azureaddfileurl=os.getenv("AZUREADDFILE")
    containername=os.getenv("CONTAINERNAME")
    azureblobnameurl=os.getenv("AZUREBLOBNAME")
except Exception as e:
    log.info("Failed at azure loading")

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
          
class Multimodal:
    
    def encode_image(self,image):
        '''Encodes image using Base64 encoding'''
        try:
            im = Image.open(image) # for testing image path
            buffered = BytesIO()
            # im.save(buffered, format="JPEG")
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
        except Exception as e:
            log.info("Failed at encode_image")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

    
    def config(self,messages,modelName):
        if modelName == "gpt4O":
            AZURE_API_KEY = os.getenv('OPENAI_API_KEY_GPT4_O')
            AZURE_API_BASE =  os.getenv('OPENAI_API_BASE_GPT4_O')            
            AZURE_API_VERSION = os.getenv('OPENAI_API_VERSION_GPT4_O')
            deployment_name = os.getenv("OPENAI_MODEL_GPT4_O")

        # client = AzureChatOpenAI(
        #                 azure_endpoint=AZURE_API_BASE,
        #                 api_key=AZURE_API_KEY,
        #                 api_version=AZURE_API_VERSION
        #             )
        try:
            # response = client.completions.create(
            #         model=deployment_name,
            #         messages=messages,
            #         max_tokens=500)
            llm = AzureChatOpenAI(deployment_name=os.getenv("OPENAI_MODEL_GPT4_O"), openai_api_version=os.getenv('OPENAI_API_VERSION_GPT4_O'), openai_api_key=os.getenv('OPENAI_API_KEY_GPT4_O'), openai_api_base=os.getenv('OPENAI_API_BASE_GPT4_O'))
            ai_message = llm.invoke(messages)
            # return json.loads(response.choices[0].message.content)
            print(ai_message.content)
            output = [ai_message.content]
            print("outputoutput",output)
            return output
        
        except Exception as e:
            log.info("Failed at image_config")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")



    # def image_rag(self,payload):
    #     '''Implements image explainability using GPT-4o
    #     Args: Prompt, Image
    #     Return: response text'''
    #     try:
    #         payload=AttributeDict(payload)
    #         text=payload.text
    #         file=payload.file
    #         # print("filefilefile", file.file)
    #         base64_image=self.encode_image(file.file)
    #         messages = [{"role": "user", "content": 
    #                     [
    #                     {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    #                     ]}]
            
    #         messages[0]["content"].append({"type": "text", "text": text})
            
    #         # messages[0]["content"].append({"type": "text", "text": template[payload['TemplateName']].replace("prompt",payload['Prompt'])})
            
    #         return self.config(messages,"gpt4O")
        
    
    #     except Exception as e:
    #         log.info("Failed at image_rag")
    #         log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
    
    def image_rag(self,payload):
        '''Implements image explainability using GPT-4o
        Args: Prompt, Image
        Return: response text'''
        try:  
            payload = AttributeDict(payload) 
            text = payload.text  
            files = payload.file  
              
            messages = [{  
                "role": "user",  
                "content": []  
            }]  
              
            for file in files:
                contents = file.file
                filename = file.filename  
                response = requests.post(url=azureaddfileurl, files={"file": (filename, contents)}, data={"container_name": containername}, headers=None, verify=False)
                if response.status_code == 200:
                    blobname_output = response.json()["blob_name"]
                    log.info(f"File uploaded successfully. Blob name: {blobname_output}, Container name: {containername}")
                else:
                    log.info(f"Error uploading file': {response.status_code} - {response.text}")
                base64_image = self.encode_image(file.file)  
                messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})  
              
            messages[0]["content"].append({"type": "text", "text": text})  
              
            response = self.config(messages, "gpt4O")  
            return response 
        
    
        except Exception as e:
            log.info("Failed at image_rag")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            