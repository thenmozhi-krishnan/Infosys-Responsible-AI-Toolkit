'''
Copyright 2024 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import math
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer,util
from detoxify import Detoxify
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from werkzeug.exceptions import InternalServerError
from fastapi.encoders import jsonable_encoder
import traceback
from mapper.mapper import *
import time
import contextvars
from config.logger import CustomLogger,request_id_var
try:
    log=CustomLogger()
    print("before loading model")
    log = CustomLogger()
    request_id_var = contextvars.ContextVar("request_id_var")
    #pipe = StableDiffusionPipeline.from_pretrained('/model/stablediffusion/fp32/model')
    device = "cuda"
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    analyzer_engine = AnalyzerEngine(registry=registry)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gpu=0 if torch.cuda.is_available() else -1

    check_point = 'toxic_debiased-c7548aa0.ckpt'
    toxicityModel = Detoxify(checkpoint='../models/detoxify/'+ check_point,
                            device=device,
                            huggingface_config_path='../models/detoxify')
    tokenizer = AutoTokenizer.from_pretrained("../models/detoxify")

    PromptModel_dberta = AutoModelForSequenceClassification.from_pretrained("../models/dbertaInjection").to(device)
    Prompttokens_dberta = AutoTokenizer.from_pretrained("../models/dbertaInjection")

    #topictokenizer_Facebook = AutoTokenizer.from_pretrained("../models/facebook")
    #topicmodel_Facebook = AutoModelForSequenceClassification.from_pretrained("../models/facebook").to(device)
    topictokenizer_dberta = AutoTokenizer.from_pretrained("../models/restricted-dberta-large-zeroshot")
    topicmodel_dberta = AutoModelForSequenceClassification.from_pretrained("../models/restricted-dberta-large-zeroshot").to(device)

    # classifier = pipeline("zero-shot-classification",model="../../models/facebook",device=device)
    # classifier2 = pipeline("zero-shot-classification",model="../../models/restricted-dberta-large-zeroshot",device=device)
    encoder = SentenceTransformer("../models/multi-qa-mpnet-base-dot-v1").to(device)
    jailbreakModel = encoder
    similarity_model =encoder
    request_id_var.set("Startup")
    log_dict={}
    print("model loaded")

except Exception as e:
    log.error(f"Exception: {e}")
    log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")

#@mp.profile()
def privacy(id,text,PIIenities_selected=None):  
    try:
        st = time.time()
        analyzer_results = analyzer_engine.analyze(text=text, language="en",entities=PIIenities_selected)
        entityList= []
        anyz_res = jsonable_encoder(analyzer_results)
        for i in anyz_res:
            entityList.append(i['entity_type'])
        et = time.time()
        rt = et-st
        return anyz_res,jsonable_encoder(entityList), {"time_taken":str(round(rt,3))+"s"}
    except Exception as e:
             
            log.error("Error occured in privacy")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at privacy call"})
            raise InternalServerError()
            
#@mp.profile()
def multi_q_net_similarity(id,text1=None,text2=None,emb1=None,emb2=None):
    try:
        st = time.time()
        if text1:
            with torch.no_grad():
                emb1 = jailbreakModel.encode(text1, convert_to_tensor=True,device=device)
        if text2:
            with torch.no_grad():
                emb2 = jailbreakModel.encode(text2, convert_to_tensor=True,device=device)
        
        emb = util.pytorch_cos_sim(emb1, emb2).to("cpu").numpy().tolist()
        del emb1
        del emb2
        et = time.time()
        rt =et-st
        torch.cuda.empty_cache()
        return emb,{'time_taken': str(round(rt,3))+"s"}
    except Exception as e:
             
            log.error("Error occured in multi_q_net_similarity")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at multi_q_net_similarity call"})
            raise InternalServerError()
            
#@mp.profile()
def multi_q_net_embedding(id,lst):
    try:
        st = time.time()
        res = []
        for text in lst:
            with torch.no_grad():
                text_embedding = jailbreakModel.encode(text, convert_to_tensor=True,device=device)
            res.append(text_embedding.to("cpu").numpy().tolist())

        del text_embedding
        et = time.time()
        rt = et-st
        
        torch.cuda.empty_cache()
        return res,{'time_taken': str(round(rt,3))+"s"}
        # return text_embedding.numpy().tolist()
    except Exception as e:
             
            log.error("Error occured in multi_q_net text embedding")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at multi_q_net text embedding call"})
            raise InternalServerError()
            
#@mp.profile()
def restricttopic_check(payload,id): 
    try:
        st = time.time()
        # topicmodel = topicmodel_Facebook
        # topictokenizer = topictokenizer_Facebook

        # nlp = pipeline('zero-shot-classification', model=classifier, tokenizer=topictokenizer)

        text=payload['text']
        labels=payload['labels']
        
        #commented to use just dberta model
        #model =payload['model'] if hasattr(payload, 'model') else "facebook"
        #if model==None:
        #    model="dberta"
        
        # if model=="facebook":
        #     # nlp = classifier
        #     nlp = pipeline('zero-shot-classification', model=topicmodel_Facebook, tokenizer=topictokenizer_Facebook, device=gpu)
        # elif model=="dberta": 
        #     # nlp = classifier2
        #     nlp = pipeline('zero-shot-classification', model=topicmodel_dberta, tokenizer=topictokenizer_dberta,device=gpu)
        nlp = pipeline('zero-shot-classification', model=topicmodel_dberta, tokenizer=topictokenizer_dberta,device=gpu)
        with torch.no_grad():    
            output=nlp(text, labels,multi_label=True)
        for i in range(len(output["scores"])):
            output["scores"][i] = round(output["scores"][i],4)

        del nlp
        torch.cuda.empty_cache()
        et = time.time()
        rt = et-st
        output['time_taken'] = str(round(rt,3))+"s"
        return output
    
    except Exception as e:
             
            log.error("Error occured in restricttopic_check")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at restricttopic_check call"})
            raise InternalServerError()

#@mp.profile()
def toxicity_check(payload,id) :
    try:
        st = time.time()
        text = payload['text']

        #to check number of tokens 
        input_ids = tokenizer.encode(text)
        result_list=[]
        #to send max 511 tokens to the model at a time and at end find avg result for each token set
        if len(input_ids)>511:
            val=math.ceil(len(input_ids)/511)
            j=0
            k=511
            for i in range(0,val):
                text="".join(tokenizer.decode(input_ids[j:k]))
                text=text.lstrip("<s>").rstrip("</s>")
                j+=511
                k+=511
                with torch.no_grad():
                    result = toxicityModel.predict(text)
                    result_list.append(result)
            output = {
                'toxicity': 0,
                'severe_toxicity': 0,
                'obscene': 0,
                'threat': 0,
                'insult': 0,
                'identity_attack': 0,
                'sexual_explicit': 0
                }
            for j in result_list:
                 output['toxicity']+=j['toxicity']
                 output['severe_toxicity']+=j['severe_toxicity']
                 output['obscene']+=j['obscene']
                 output['identity_attack']+=j['identity_attack']
                 output['insult']+=j['insult']
                 output['threat']+=j['threat']
                 output['sexual_explicit']+=j['sexual_explicit']
            output = {k: v / len(result_list) for k, v in output.items()}
        else:
            with torch.no_grad():
                output = toxicityModel.predict(text)
        List_profanity_score = []
        obj_profanityScore_toxic = profanityScore(metricName='toxicity',
                                        metricScore=output['toxicity'])
        obj_profanityScore_severe_toxic = profanityScore(metricName='severe_toxicity',
                                        metricScore=output['severe_toxicity'])
        obj_profanityScore_obscene = profanityScore(metricName='obscene',
                                        metricScore=output['obscene'])
        obj_profanityScore_threat = profanityScore(metricName='threat',
                                        metricScore=output['threat'])
        obj_profanityScore_insult = profanityScore(metricName='insult',
                                        metricScore=output['insult'])
        obj_profanityScore_identity_attack = profanityScore(metricName='identity_attack',
                                        metricScore=output['identity_attack'])
        obj_profanityScore_sexual_explicit = profanityScore(metricName='sexual_explicit',
                                        metricScore=output['sexual_explicit'])
        
        List_profanity_score.append(obj_profanityScore_toxic)
        List_profanity_score.append(obj_profanityScore_severe_toxic)
        List_profanity_score.append(obj_profanityScore_obscene)
        List_profanity_score.append(obj_profanityScore_threat)
        List_profanity_score.append(obj_profanityScore_insult)
        List_profanity_score.append(obj_profanityScore_identity_attack)
        List_profanity_score.append(obj_profanityScore_sexual_explicit)

        objProfanityAnalyzeResponse = {}
        objProfanityAnalyzeResponse['toxicScore'] = List_profanity_score
        et = time.time()

        rt = et-st
        objProfanityAnalyzeResponse['time_taken'] = str(round(rt,3))+"s"
        torch.cuda.empty_cache()
        return objProfanityAnalyzeResponse
    
    except Exception as e:
             
            log.error("Error occured in toxicity_check")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at toxicity_check call"})
            raise InternalServerError()
            
#@mp.profile()    
def promptInjection_check(text,id):
    try:
        st = time.time()
        Prompttokens = Prompttokens_dberta
        PromptModel = PromptModel_dberta

        tokens = Prompttokens.encode_plus(text, truncation=True, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = PromptModel(**tokens)

        predicted_label = outputs.logits.argmax().item()
        label_names = PromptModel.config.id2label
        predicted_label_name = label_names[predicted_label]
        predicted_probabilities = outputs.logits.softmax(dim=1)[0, predicted_label].item()

        del tokens
        et = time.time()
        rt = et-st

        torch.cuda.empty_cache()
        # #torch.cuda.empty_cache()
        return predicted_label_name,predicted_probabilities, {'time_taken':str(round(rt,3))+"s"}
    except Exception as e:
            
            log.error("Error occured in promptInjection_check")
            log.error(f"Exception: {str(traceback.extract_tb(e.__traceback__)[0].lineno),e}")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at promptInjection_check call"})
            raise InternalServerError()
