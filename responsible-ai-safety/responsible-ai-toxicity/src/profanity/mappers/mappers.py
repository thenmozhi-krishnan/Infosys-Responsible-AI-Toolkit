"""
 <2023> Infosys Limited, Bangalore, India. All Rights Reserved.
 Version: 
Except for any free or open source software components embedded in this Infosys proprietary software program ( Program ), 
this Program is protected by copyright laws, international treaties and other pending or existing intellectual property rights in India, 
the United States and other countries. Except as expressly permitted, any unauthorized reproduction, storage, 
transmission in any form or by any means (including without limitation electronic, mechanical, printing, photocopying, recording or otherwise), 
or any distribution of this Program, or any portion of it, may result in severe civil and criminal penalties, 
and will be prosecuted to the maximum extent possible under the law.
"""

"""
fileName: mappers.py
description: A Pydantic model object for usecase entity model
             which maps the data model to the usecase entity schema
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Union, List

"""
description: profinityEntity
params:
"""


class profanity(BaseModel):
    profaneWord: str = Field(example="dummy")
    beginOffset: int = Field(example=9)
    endOffset: int = Field(example=14)

    class Config:
        orm_mode = True

class profanityScoreList(BaseModel):
    metricName: str = Field(example="toxicity")
    metricScore: float = Field(example=0.78326)

    class Config:
        orm_mode = True

class ProfanityAnalyzeRequest(BaseModel):
    inputText: str = Field(example="You are a dummy")
    user: Optional[str] = Field(None)
    lotNumber: Optional[str] = Field(None)


class ProfanityAnalyzeResponse(BaseModel):
    profanity: List[profanity]
    profanityScoreList: List[profanityScoreList]

    class Config:
        orm_mode = True


class ProfanitycensorRequest(BaseModel):
    inputText: str = Field(example="You are a dummy")
    user: Optional[str] = Field(None)
    lotNumber: Optional[str] = Field(None)

class ProfanitycensorResponse(BaseModel):
    outputText: str = Field(example="You are a ****")

    class Config:
        orm_mode = True
