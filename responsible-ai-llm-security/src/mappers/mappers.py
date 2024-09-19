'''
MIT license https://opensource.org/licenses/MIT
Copyright 2024 Infosys Ltd
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from pydantic import BaseModel
from typing import Union
from fastapi import UploadFile

class AddLLMModel(BaseModel):
     modelDetails : dict ={ "modelName":"google/gemma-7b",
     "modelDeploymentEndPoint" :"https://api-inference.huggingface.co/models/google/gemma-7b",
     "headers":{
        'Content-Type': 'application/json'
    },
     "payload": {
        "inputs": ["prompt"],
        "parameters": {
            "max_new_tokens": 100
        }
    }
     
     }
     

     
class DeleteLLMModel(BaseModel):
     modelName: str ="chatgpt"

class AddDataset(BaseModel):
     datasetName: str ="sst2"
     datasetFile : Union[UploadFile,None]=None

class DeleteLLMDataset(BaseModel):
     datasetName: str ="sst2"

class LLMRobustness(BaseModel):
     modelName: str = "google/flan-t5-large"
     dataset: str = "sst2"
     numberOfSamples: int= 5
     prompts : list =["Classify the sentence as positive or negative: {content}",
                            "Determine the emotion of the following sentence as positive or negative: {content}"
                            ]

class LLMDataset(BaseModel):
     datasetName : str ="sst2"
     numberOfEntries : int = 5


class LLMResults(BaseModel):
     resultId : str = "123456"

class AddExternalRobustnessScore(BaseModel):

     RobustnessScore: dict ={
          "modelName":"T5-Large",
          "SST_2":"0.04±0.11",
          "CoLA": "0.16±0.19",
          "QQP":"0.09±0.15",
          "MPRC": "0.17±0.26",
          "MNLI":"0.08±0.13",
          "QNLI":"0.33±0.25",
          "RTE":"0.08±0.13",
          "WNLI":"0.13±0.14",
          "MMLU":"0.11±0.18",
          "SQuAD_v2":"0.05±0.12",
          "IWSLT":"0.14±0.17",
          "UN_Multi":"0.13±0.14",
          "Math":"0.24±0.21",
          "Avg":"0.13±0.19",
          "inhouse_model":True
     }

class AddExternalAttackScore(BaseModel):
     
     attackScore: dict = {
          "modelName":"T5-Large",
          "TextBugger":"0.13±0.18",
          "DeepWordBug":"0.20±0.24",
          "TextFoller":"0.21±0.24",
          "BertAttack":"0.04±0.08",
          "CheckList":"0.18±0.24",
          "StressTest":"0.18±0.24",
          "Semantic":"0.10±0.09",
          "inhouse_model":True
     }


