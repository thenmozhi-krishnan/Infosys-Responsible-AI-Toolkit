'''
MIT license https://opensource.org/licenses/MIT
Copyright 2024 Infosys Ltd
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from mappers.mappers import LLMRobustness,LLMDataset,AddLLMModel,AddDataset,LLMResults,DeleteLLMModel,DeleteLLMDataset,AddExternalRobustnessScore,AddExternalAttackScore
from service.service import LLMs
from fastapi import APIRouter,Query
from fastapi import Depends
from fastapi.responses import FileResponse

model = APIRouter()
datasets=APIRouter()
robustness=APIRouter()
results=APIRouter()


@model.get('/v1/infosys/llm/security/models')
def availabelModels():
    return LLMs.getModels()

@model.post('/v1/infosys/llm/security/addModel')
def newModel(payload:AddLLMModel=Depends()):
    return LLMs.addModel(payload)

@model.post('/v1/infosys/llm/security/deleteModel')
def deleteModel(payload:DeleteLLMModel=Depends()):
    return LLMs.modelDelete(payload)

@datasets.get('/v1/infosys/llm/security/datasets')
def availabelDatasets():
    return LLMs.getDatasets()

@datasets.post('/v1/infosys/llm/security/dataset/preview')
def availabelDatasetsValues(payload : LLMDataset = Depends()):
    return LLMs.getDatasetsGlimpse(payload)

@datasets.post('/v1/infosys/llm/security/addDataset')
def newDataset(payload:AddDataset = Depends()):
    return LLMs.addDataset(payload)


@datasets.post('/v1/infosys/llm/security/deleteDataset')
def deleteDataset(payload:DeleteLLMDataset=Depends()):
    return LLMs.datasetDelete(payload)

@robustness.post('/v1/infosys/llm/security/robustness')
def calculateRobustness(payload : LLMRobustness=Depends()):
    return LLMs.getRobustness(payload)

@robustness.get('/v1/infosys/llm/security/leaderBoard')
def LLMLeaderboard():
    return LLMs.getLeaderboard()

@robustness.post('/v1/infosys/llm/security/robustness/results')
def LLMResults(payload:LLMResults=Depends()):
    return LLMs.results(payload)


@results.post('/v1/infosys/llm/security/addExternalRobustnessScore')
def addLLMEXternalRobustnessScore(payload:AddExternalRobustnessScore):
    return LLMs.addExternalRobustnessScore(payload)

@results.post('/v1/infosys/llm/security/addExternalAttackScore')
def addLLMExternalAttackScore(payload:AddExternalAttackScore):
    return LLMs.addExternalAttackScore(payload)

@results.get('/v1/infosys/llm/security/getExternalRobustnessScore')
def getLLMExternalRobustnessScore():
    return LLMs.getExternalRobustnessScore()

@results.get('/v1/infosys/llm/security/getExternalAttackScore')
def getLLMExternalAttackScore():
    return LLMs.getExternalAttackScore()

@results.post('/v1/infosys/llm/security/deleteExternalAttackScore')
def deleteLLMExternalAttackScore(payload:AddExternalAttackScore):
    return LLMs.deleteExternalAttackScore(payload)

@results.post('/v1/infosys/llm/security/deleteExternalRobustnessScore')
def deleteLLMExternalRobustnessScore(payload:AddExternalRobustnessScore):
    return LLMs.deleteExternalRobustnessScore(payload)

