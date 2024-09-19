from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer,util
from detoxify import Detoxify
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from modeldeployment.mapper.mapper import *
from modeldeployment.config.logger import CustomLogger, request_id_var
import torch
import traceback
from modeldeployment.service.utils import perplexity, entropy
log = CustomLogger()
request_id_var.set("Startup")
try:
    global log_dict
    log_dict={}
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    analyzer_engine = AnalyzerEngine(registry=registry)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gpu=0 if torch.cuda.is_available() else -1
    check_point = 'toxic_debiased-c7548aa0.ckpt'
    # toxicityModel = Detoxify(checkpoint='../models/detoxify/'+ check_point,
    #                         device=device,
    #                         huggingface_config_path='../models/detoxify')

    # PromptModel_dberta = AutoModelForSequenceClassification.from_pretrained("../models/dbertaInjection").to(device)
    # Prompttokens_dberta = AutoTokenizer.from_pretrained("../models/dbertaInjection")

    # topictokenizer_Facebook = AutoTokenizer.from_pretrained("../models/facebook")
    # topicmodel_Facebook = AutoModelForSequenceClassification.from_pretrained("../models/facebook").to(device)

    # topictokenizer_dberta = AutoTokenizer.from_pretrained("../models/restricted-dberta-large-zeroshot")
    # topicmodel_dberta = AutoModelForSequenceClassification.from_pretrained("../models/restricted-dberta-large-zeroshot").to(device)

    # # classifier = pipeline("zero-shot-classification",model="../models/facebook",device=device)
    # # classifier2 = pipeline("zero-shot-classification",model="../models/restricted-dberta-large-zeroshot",device=device)
    # encoder = SentenceTransformer("../models/multi-qa-mpnet-base-dot-v1").to(device)
    # jailbreakModel = encoder
    # similarity_model =encoder
    print("---------Loading Falcon-7b--------------")
    observerModel_falcon = AutoModelForCausalLM.from_pretrained("../models/falcon-7b").to(device)
    print("---------Loading Falcon-7b-instruct--------------")
    performerModel_falcon = AutoModelForCausalLM.from_pretrained("../models/falcon-7b-instruct").to(device)
    observerTokens_falcon = AutoTokenizer.from_pretrained("../models/falcon-7b")
    print("---------Falcon-7b Loading Finished--------------")


    
    
except Exception as e:
    log.error(f"Exception: {e}")

def privacy(text,PIIenities_selected=None):  
    try:
        analyzer_results = analyzer_engine.analyze(text=text, language="en",entities=PIIenities_selected)
        entityList= []
        for i in analyzer_results:
            entityList.append(i.entity_type)
        return analyzer_results,entityList
    except Exception as e:
            log.error("Error occured in privacy")
            log.error(f"Exception: {e}")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at privacy call"})
            

def multi_q_net_similarity(text1=None,text2=None,emb1=None,emb2=None):
    try:
        if text1:
            with torch.no_grad():
                emb1 = jailbreakModel.encode(text1, convert_to_tensor=True,device=device)
        if text2:
            with torch.no_grad():
                emb2 = jailbreakModel.encode(text2, convert_to_tensor=True,device=device)
        
        emb = util.pytorch_cos_sim(emb1, emb2).to("cpu").numpy().tolist()
        del emb1
        del emb2
        torch.cuda.empty_cache()
        return emb
    except Exception as e:
            log.error("Error occured in multi_q_net_similarity")
            log.error(f"Exception: {e}")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at multi_q_net_similarity call"})
            

def multi_q_net_embedding(lst):
    try:
        res = []
        for text in lst:
            with torch.no_grad():
                text_embedding = jailbreakModel.encode(text, convert_to_tensor=True,device=device)
            res.append(text_embedding.to("cpu").numpy().tolist())

        del text_embedding
        torch.cuda.empty_cache()
        return res
        # return text_embedding.numpy().tolist()
    except Exception as e:
            log.error("Error occured in multi_q_net text embedding")
            log.error(f"Exception: {e}")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at multi_q_net text embedding call"})
            

def restricttopic_check(payload): 
    try:
        # topicmodel = topicmodel_Facebook
        # topictokenizer = topictokenizer_Facebook

        # nlp = pipeline('zero-shot-classification', model=classifier, tokenizer=topictokenizer)

        text=payload.text
        labels=payload.labels
        
        model =payload.model if hasattr(payload, 'model') else "facebook"
        if model==None:
            model="dberta"
        
        if model=="facebook":
            # nlp = classifier
            nlp = pipeline('zero-shot-classification', model=topicmodel_Facebook, tokenizer=topictokenizer_Facebook, device=gpu)
        elif model=="dberta": 
            # nlp = classifier2
            nlp = pipeline('zero-shot-classification', model=topicmodel_dberta, tokenizer=topictokenizer_dberta,device=gpu)
        with torch.no_grad():    
            output=nlp(text, labels,multi_label=True)
        for i in range(len(output["scores"])):
            output["scores"][i] = round(output["scores"][i],4)

        del nlp
        torch.cuda.empty_cache()
        return output
    
    except Exception as e:
            log.error("Error occured in restricttopic_check")
            log.error(f"Exception: {e}")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at restricttopic_check call"})

def toxicity_check(payload: detoxifyRequest) -> detoxifyResponse:
    try:
        text = payload.text
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

        objProfanityAnalyzeResponse = detoxifyResponse
        objProfanityAnalyzeResponse.toxicScore = List_profanity_score

        torch.cuda.empty_cache()
        return objProfanityAnalyzeResponse
    
    except Exception as e:
            log.error("Error occured in toxicity_check")
            log.error(f"Exception: {e}")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at toxicity_check call"})
            
    
def promptInjection_check(text):
    try:

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
        torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
        return predicted_label_name,predicted_probabilities
    except Exception as e:
            log.error("Error occured in promptInjection_check")
            log.error(f"Exception: {e}")
            log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                   "Error Module":"Failed at promptInjection_check call"})
            
def textDetection_check(text, mode: str = "low-fpr"):
    print("-----------Text Detection Started--------------")
    BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
    BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

    if mode == "low-fpr":
        threshold = BINOCULARS_FPR_THRESHOLD
    elif mode == "accuracy":
        threshold = BINOCULARS_ACCURACY_THRESHOLD
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    try:
        print("Model 1 --> ", observerModel_falcon)
        print("Model 2 --> ", performerModel_falcon)
        print("Tokenizer -->", observerTokens_falcon)
        observerModel = observerModel_falcon
        performerModel = performerModel_falcon
        tokenizer = observerTokens_falcon
        print("---------Initialization Finished-------------")
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_token_type_ids=False).to(device)
        print("-------------Generating Logits-------------")
        with torch.no_grad():
            observer_logits = observerModel(**encodings.to(device)).logits
            performer_logits = performerModel(**encodings.to(device)).logits

        print("-------------Generating Perplexity Score-------------")
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(device), performer_logits.to(device),
                        encodings.to(device), tokenizer.pad_token_id)
        print("-----------Score Generated-----------")
        score = ppl / x_ppl
        score = score.tolist()[0]
        pred = "Most likely AI-generated" if score < threshold else "Most likely human-generated"
                       
        del encodings, observer_logits, performer_logits
        torch.cuda.empty_cache()
        print("------------Text Detection Finished--------------")
        return { "prediction": pred, "score": f"{score:.3f}", "threshold": f"{threshold:.3f}" }
    except Exception as e:
        log.error("Error occured in textDetection_check")
        log.error(f"Exception: {e}")
        log_dict[request_id_var.get()].append({"Line number":str(traceback.extract_tb(e.__traceback__)[0].lineno),"Error":str(e),
                                                "Error Module":"Failed at textDetection_check call"})



        