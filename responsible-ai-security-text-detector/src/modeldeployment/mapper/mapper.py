from pydantic import BaseModel, Field,Extra
from typing import Optional, Union, List
from enum import Enum

class detoxifyRequest(BaseModel):
    text:str= Field(example="Russia is the biggest country by area.")
   
    class Config:
        orm_mode = True
class PIICHECKS(str, Enum):
    PERSON = 'PERSON'
    LOCATION = 'LOCATION'
    DATE  = 'DATE'
    AU_ABN  = 'AU_ABN'
    AU_ACN  = 'AU_ACN'
    AADHAR_NUMBER  = 'AADHAR_NUMBER'
    AU_MEDICARE  = 'AU_MEDICARE'
    AU_TFN  = 'AU_TFN'
    CREDIT_CARD = 'CREDIT_CARD'
    CRYPTO = 'CRYPTO'
    DATE_TIME  = 'DATE_TIME'
    EMAIL_ADDRESS  = 'EMAIL_ADDRESS'
    ES_NIF  = 'ES_NIF'
    IBAN_CODE  = 'IBAN_CODE'
    IP_ADDRESS  = 'IP_ADDRESS'
    IT_DRIVER_LICENSE  = 'IT_DRIVER_LICENSE'
    IT_FISCAL_CODE = 'IT_FISCAL_CODE'
    IT_IDENTITY_CARD = 'IT_IDENTITY_CARD'
    IT_PASSPORT  = 'IT_PASSPORT'
    IT_VAT_CODE  = 'IT_VAT_CODE'
    MEDICAL_LICENSE  = 'MEDICAL_LICENSE'
    PAN_Number  = 'PAN_Number'
    PHONE_NUMBER  = 'PHONE_NUMBER'
    SG_NRIC_FIN  = 'SG_NRIC_FIN'
    UK_NHS  = 'UK_NHS'
    URL  = 'URL'
    PASSPORT  = 'PASSPORT'
    US_ITIN  = 'US_ITIN'
    US_PASSPORT  = 'US_PASSPORT'
    US_SSN  = 'US_SSN'

    class Config:
        orm_mode = True

class privacyRequest(BaseModel):
    text:str= Field(example="Russia is the biggest country by area.")
    entitiesselected:List[PIICHECKS] =Field(example=['PERSON','LOCATION','DATE','AU_ABN','AU_ACN','AADHAR_NUMBER','AU_MEDICARE','AU_TFN','CREDIT_CARD','CRYPTO','DATE_TIME','EMAIL_ADDRESS','ES_NIF','IBAN_CODE','IP_ADDRESS','IT_DRIVER_LICENSE','IT_FISCAL_CODE','IT_IDENTITY_CARD','IT_PASSPORT','IT_VAT_CODE','MEDICAL_LICENSE','PAN_Number','PHONE_NUMBER','SG_NRIC_FIN','UK_NHS','URL','PASSPORT','US_ITIN','US_PASSPORT','US_SSN'])
   
    class Config:
        orm_mode = True

class profanityScore(BaseModel):
    metricName: str = Field(example="toxicity")
    metricScore: float = Field(example=0.78326)

    class Config:
        orm_mode = True

class JailbreakRequest(BaseModel):
    text: List[str]

class detoxifyResponse(BaseModel):
    toxicScore: List[profanityScore]

    class Config:
        orm_mode = True

class model(str, Enum):
    dberta= 'dberta'
    facebook = 'facebook'

    class Config:
        orm_mode = True

class RestrictedTopicRequest(BaseModel):
    text:str= Field(example="Russia is the biggest country by area.")
    labels:List = Field(["Terrorism", "Explosives"])
    model:Optional[str] = Field(example="dberta")
    class Config:
        orm_mode = True

class SimilarityRequest(BaseModel):
    text1:Optional[str]= Field(default=None, example="Russia is the biggest country by area.")
    text2:Optional[str]= Field(default=None, example="Russia is the biggest country by area.")
    emb1:Optional[List] =None
    emb2:Optional[List] =None
    class Config:
        orm_mode = True





    


