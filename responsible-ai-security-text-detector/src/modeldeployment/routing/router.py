from fastapi import Depends,Request,APIRouter, HTTPException, Header, Body
from modeldeployment.mapper.mapper import *
# from modeldeployment.service.service import Toxicity as service
from modeldeployment.service.service import *
import time
import uuid
from modeldeployment.exception.exception import modeldeploymentException
# from modeldeployment.dao.AdminDb import Results
from modeldeployment.config.logger import CustomLogger, request_id_var
from modeldeployment.service.service import log_dict
from typing import Dict, Any
import traceback
import uuid

router = APIRouter()
log=CustomLogger()
# @router.post("/models/detoxifymodel", response_model=detoxifyResponse)
# def toxic_model(payload: detoxifyRequest):
#     st=time.time()
#     log.info("Entered create usecase routing method")
#     try:
#         log.info("before invoking create usecase service ")
#         id=uuid.uuid4().hex
#         request_id_var.set(id)
#         log_dict[request_id_var.get()]=[]
#         response = toxicity_check(payload)
#         log.info("after invoking create usecase service ")
        
#         er=log_dict[request_id_var.get()]
#         logobj = {"_id":id,"error":er}
#         # if len(er)!=0:
#             # Results.createlog(logobj)
#         del log_dict[id]
#         log.debug("response : " + str(response))
#         log.info("exit create usecase routing method")
#         log.info(f"Time taken by toxicity {time.time()-st}")
#         return response
#     except modeldeploymentException as cie:
#         log.error(cie.__dict__)
#         log.info("exit create usecase routing method")
#         raise HTTPException(**cie.__dict__)
    
# @router.post("/models/privacy")
# def pii_check(payload: privacyRequest):
#     st=time.time()
#     log.info("Entered create usecase routing method")
#     try:
#         log.info("before invoking create usecase service ")
#         id=uuid.uuid4().hex
#         request_id_var.set(id)
#         log_dict[request_id_var.get()]=[]
#         response = privacy(payload.text,payload.entitiesselected)
#         log.info("after invoking create usecase service ")
#         er=log_dict[request_id_var.get()]
#         logobj = {"_id":id,"error":er}
#         # if len(er)!=0:
#         #     Results.createlog(logobj)
#         del log_dict[id]
#         log.debug("response : " + str(response))
#         # log.debug("response : " + str(response))
#         log.info("exit create usecase routing method")
#         log.info(f"Time taken by privacy {time.time()-st}")
#         return response
#     except modeldeploymentException as cie:
#         log.error(cie.__dict__)
#         log.info("exit create usecase routing method")
#         raise HTTPException(**cie.__dict__)
    
# @router.post("/models/promptinjectionmodel")
# def prompt_model(payload: detoxifyRequest):
#     st=time.time()
#     log.info("Entered create usecase routing method")
#     try:
#         log.info("before invoking create usecase service ")
#         id=uuid.uuid4().hex
#         request_id_var.set(id)
#         log_dict[request_id_var.get()]=[]
#         response = promptInjection_check(payload.text)
#         log.info("after invoking create usecase service ")
#         er=log_dict[request_id_var.get()]
#         logobj = {"_id":id,"error":er}
#         # if len(er)!=0:
#         #     Results.createlog(logobj)
#         del log_dict[id]
#         log.debug("response : " + str(response))
#         # log.debug("response : " + str(response))
#         log.info("exit create usecase routing method")
#         log.info(f"Time taken by promptinjection {time.time()-st}")
#         return response
#     except modeldeploymentException as cie:
#         log.error(cie.__dict__)
#         log.info("exit create usecase routing method")
#         raise HTTPException(**cie.__dict__)
    
# @router.post("/models/restrictedtopicmodel")
# def restrictedTopic_model(payload: RestrictedTopicRequest):
#     st=time.time()
#     log.info("Entered create usecase routing method")
#     try:
#         log.info("before invoking create usecase service ")
#         id=uuid.uuid4().hex
#         request_id_var.set(id)
#         log_dict[request_id_var.get()]=[]
#         response = restricttopic_check(payload)
#         log.info("after invoking create usecase service ")
#         er=log_dict[request_id_var.get()]
#         logobj = {"_id":id,"error":er}
#         # if len(er)!=0:
#         #     Results.createlog(logobj)
#         del log_dict[id]
#         log.debug("response : " + str(response))
#         # log.debug("response : " + str(response))
#         log.info("exit create usecase routing method")
#         log.info(f"Time taken by RestrictedTopic{time.time()-st}")
#         return response
#     except modeldeploymentException as cie:
#         log.error(cie.__dict__)
#         log.info("exit create usecase routing method")
#         raise HTTPException(**cie.__dict__)
    
# @router.post("/models/multi_q_net_embedding")
# def embedding_model(payload: JailbreakRequest):
#     st=time.time()
#     log.info("Entered create usecase routing method")
#     try:
#         log.info("before invoking create usecase service ")
#         id=uuid.uuid4().hex
#         request_id_var.set(id)
#         log_dict[request_id_var.get()]=[]
#         response = multi_q_net_embedding(payload.text)
#         log.info("after invoking create usecase service ")
#         er=log_dict[request_id_var.get()]
#         logobj = {"_id":id,"error":er}
#         # if len(er)!=0:
#         #     Results.createlog(logobj)
#         del log_dict[id]
#         log.debug("response : " + str(response))
#         # log.debug("response : " + str(response))
#         log.info("exit create usecase routing method")
#         log.info(f"Time taken by Jailbreak {time.time()-st}")
#         return response
#     except modeldeploymentException as cie:
#         log.error(cie.__dict__)
#         log.info("exit create usecase routing method")
#         raise HTTPException(**cie.__dict__)

# @router.post("/models/multi-qa-mpnet-model_similarity")
# def similarity_model(payload: SimilarityRequest,authorization: str = Header(None)):
#     st=time.time()
#     log.info("Entered create usecase routing method")
#     try:
#         log.info("before invoking create usecase service ")
#         id=uuid.uuid4().hex
#         request_id_var.set(id)
#         log_dict[request_id_var.get()]=[]
#         response = multi_q_net_similarity(payload.text1,payload.text2,payload.emb1,payload.emb2)
#         log.info("after invoking create usecase service ")
#         er=log_dict[request_id_var.get()]
#         logobj = {"_id":id,"error":er}
#         # if len(er)!=0:
#         #     Results.createlog(logobj)
#         del log_dict[id]
#         log.debug("response : " + str(response))
#         # log.debug("response : " + str(response))
#         log.info("exit create usecase routing method")
#         log.info(f"Time taken by similary{time.time()-st}")
#         return response
#     except modeldeploymentException as cie:
#         log.error(cie.__dict__)
#         log.info("exit create usecase routing method")
#         raise HTTPException(**cie.__dict__)

@router.post("/models/text-detection")
def textDetection_model(payload: detoxifyRequest):
    st=time.time()
    log.info("Entered create usecase routing method")
    try:
        log.info("before invoking create usecase service ")
        id=uuid.uuid4().hex
        request_id_var.set(id)
        log_dict[request_id_var.get()]=[]
        response = textDetection_check(payload.text)
        log.info("after invoking create usecase service ")
        
        er=log_dict[request_id_var.get()]
        logobj = {"_id":id,"error":er}
        # if len(er)!=0:
        #     Results.createlog(logobj)
        del log_dict[id]
        log.debug("response : " + str(response))
        log.info("exit create usecase routing method")
        log.info(f"Time taken by textDetection {time.time()-st}")
        return response
    except modeldeploymentException as cie:
        log.error(cie.__dict__)
        log.info("exit create usecase routing method")
        raise HTTPException(**cie.__dict__)