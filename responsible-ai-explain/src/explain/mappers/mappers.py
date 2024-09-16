'''
Copyright 2024 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies 
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE 
AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

"""
fileName: mappers.py
description: A Pydantic model object for usecase entity model
             which maps the data model to the usecase entity schema
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum

"""
description: Explainability
params:
"""

class DatasetType(str, Enum):
    csv = 'text/csv'
    parquet = 'text/parquet'

class TaskType(str, Enum):
    CLASSIFICATION  = 'CLASSIFICATION'
    REGRESSION = 'REGRESSION'
    TIMESERIESFORECAST = 'TIMESERIESFORECAST'

class LocalExplainabilityTabularMethods(str, Enum):
    ANCHOR_TABULAR = 'ANCHOR-TABULAR'
    KERNEL_SHAP = 'KERNEL-SHAP'
    TREE_SHAP = 'TREE-SHAP'
    TS_LIMEEXPLAINER= 'TS-LIMEEXPLAINER'
    LIME_TABULAR = 'LIME-TABULAR'

class GlobalExplainabilityMethods(str, Enum):
    PD_VARIANCE = 'PD-VARIANCE'
    KERNEL_SHAP = 'KERNEL-SHAP'
    TREE_SHAP = 'TREE-SHAP'
    PERMUTATION_IMPORTANCE = 'PERMUTATION-IMPORTANCE'
    PARTIAL_DEPENDENCE = 'PARTIAL-DEPENDENCE'

class LocalExplainabilityTextMethods(str, Enum):
    ANCHOR_TEXT = 'ANCHOR-TEXT'
    INTEGRATED_GRADIENTS = 'INTEGRATED-GRADIENTS'

class LocalExplainabilityImageMethods(str, Enum):
    ANCHOR_IMAGE = 'ANCHOR-IMAGE'
    INTEGRATED_GRADIENTS = 'INTEGRATED-GRADIENTS'
    LIME = 'LIME'

class SegmentationType(str, Enum):
    # Anchor Segmentation Types
    felzenszwalb = 'felzenszwalb'
    slic = 'slic'
    quickshift = 'quickshift'
    
    # Integrated Gradient Segmentation Types
    gausslegendre = 'gausslegendre'
    riemann_left = 'riemann_left'
    riemann_right = 'riemann_right'
    riemann_middle = 'riemann_middle'
    riemann_trapezoid = 'riemann_trapezoid'

class importantFeatures(BaseModel):
    name: str = Field(example="sex")
    value: float = Field(example=0.9)
    
class ExplainabilityLocalTabular(BaseModel):
    modelPrediction: Optional[str] = Field(example="Class_GOOD")
    modelConfidence: Optional[float] = Field(example= "0.85")
    anchor: Optional[List] = Field(example=["awesome"])
    importantFeatures : Optional[List[importantFeatures]]
    limeTimeSeries : Optional[str]  = Field(example="base64 explain image")
    inputRow: Optional[List] = Field(example=[{}])
    class Config:
        from_attributes = True

class ExplainabilityLocalDemoRequest(BaseModel):
    inputText: str = Field(example="Unfortunately the movie served with bad visuals")
    explainerID: int = Field(example=1)
    portfolio:Optional[str] = Field(example="string")
    account:Optional[str] = Field(example="string")
    user: Optional[str] = Field(example = "admin")
    lotNumber:Optional[str] = Field(example = "p1")

    class Config:
        from_attributes = True

#this is for demo dont remove
class explainabilitylocal(BaseModel):
    predictedTarget: str = Field(example="Class_GOOD")
    anchor: List = Field(example=["awesome"])
    explanation: Optional[str] = Field(example="string")
 
    class Config:
        from_attributes = True
        
class ExplainabilityLocalResponseDemo(BaseModel):
    explainerID: int = Field(example=1)
    explanation: List[explainabilitylocal]
    class Config:
        from_attributes = True
  
class ExplainabilityLocalTabularResponse(BaseModel):
    explainerID: int = Field(example=1)
    explanation: List[ExplainabilityLocalTabular]
    class Config:
        from_attributes = True

class ExplainabilityLocalTabularRequest(BaseModel):
    explainerID: int = Field(example=1)
    inputIndex: int = Field(example=5)
    taskType: TaskType = Field(example="CLASSIFICATION")
    methods: LocalExplainabilityTabularMethods = Field(example="KERNEL-SHAP")
    modelPredictionEndpoint: str = Field(example="responsible-ai//responsible-ai-explain//models//1//local_explain_classifier.pkl")
    preprocessorPredictionEndpoint: Optional[str] = Field(
        example="responsible-ai//responsible-ai-explain//models//1//local_explain_preprocessor.pkl")
    datasetPath: str = Field(example="responsible-ai//responsible-ai-explain//dataset//1//adult.csv")
    datasetType: DatasetType = Field(example="text/csv")
    featureNames: Optional[List] = Field(
       example=["Age", "Workclass", "Education", "Marital Status", "Occupation", "Relationship", "Race", "Sex",
                 "Capital Gain", "Capital Loss", "Hours per week", "Country"])
    categoricalFeatureNames: Optional[List] = Field(
        example=["Workclass", "Education", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Country"])
    targetNames: str = Field(example="Target")
    targetClassNames: Optional[List] = Field(example=[">50K","<=50K"])
    outputStorageType: str = Field(example="NUTANIX")
    outputType: str = Field(example="JSON")
    outputPath: str = Field(example="responsible-ai//responsible-ai-explain//output//1//localTabular_explain.json")

    class Config:
        from_attributes = True

class explainabilityglobal(BaseModel):
    importantFeatures : List[importantFeatures]

    class Config:
        from_attributes = True

class ExplainabilityGlobalResponse(BaseModel):
    explanation: List[explainabilityglobal]
    
    class Config:
        from_attributes = True

class ExplainabilityGlobalRequest(BaseModel):
    modelID: str = Field(example=36)
    taskType: TaskType = Field(example="CLASSIFICATION")
    methods: GlobalExplainabilityMethods = Field(example="PD-VARIANCE")
    modelPredictionEndpoint: str = Field(example="responsible-ai//responsible-ai-explain//models//1//global_explain_gradientboostregressor.pkl")
    preprocessorPredictionEndpoint: Optional[str] = Field(example="responsible-ai//responsible-ai-explain//models//1//global_explain_preprocessor.pkl")
    datasetPath: str = Field(example="responsible-ai//responsible-ai-explain//dataset//1//adult.csv")
    datasetType: DatasetType = Field(example="text/csv")
    featureNames: Optional[List] = Field(example=["Age", "Workclass", "Education", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country"])
    categoricalFeatureNames: Optional[List] = Field(example=["Workclass", "Education", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Country"])
    targetNames: str = Field(example="Target")
    targetClassNames: Optional[List] = Field(example=[">50K", "<=50K"])
    outputStorageType: str = Field(example="NUTANIX")
    outputType: str = Field(example="JSON")
    outputPath: str = Field(example="responsible-ai//responsible-ai-explain//output//1//global_explain.json")

    class Config:
        from_attributes = True

#today changes
class Scope(str, Enum):
    GLOBAL = "GLOBAL"
    LOCAL = "LOCAL"

class DataType(str, Enum):
    UNSTRUCTURE = "UNSTRUCTURE"
    STRUCTURE = "STRUCTURE"
    IMAGE = "IMAGE"

class Status(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"

class Methods(str, Enum):
    ANCHOR_TEXT = "ANCHOR-TEXT"
    ANCHOR_TABULAR = "ANCHOR-TABULAR"
    ANCHOR_IMAGE = "ANCHOR-IMAGE"
    KERNEL_SHAP = "KERNEL-SHAP"
    TREE_SHAP = "TREE-SHAP"
    PD_VARIANCE = "PD-VARIANCE"
    PERMUTATION_IMPORTANCE = 'PERMUTATION-IMPORTANCE'
    TS_LIMEEXPLAINER= 'TS-LIMEEXPLAINER'
    LIME_TABULAR = 'LIME-TABULAR'
    INTEGRATED_GRADIENTS = 'INTEGRATED-GRADIENTS'
    LIME = 'LIME'

class ExplainInput(BaseModel):
    scope: Scope = Field(example='GLOBAL')
    taskType: TaskType = Field(example="CLASSIFICATION")
    dataType: DataType = Field(example="STRUCTURE")
    method: Methods = Field(example='KERNEL-SHAP')
    uniqueId: str = Field(example="hex code")
    inputText: Optional[str] = Field(example="string")
    targetNames: Optional[str] = Field(example="is_attrited")
    targetClassNames: Optional[List] = Field(example=["Not Attrited","Attrited"])

    class Config:
        from_attributes = True

class ExplainFileOutput(BaseModel):
    status: Status = Field(examples=['SUCCESS','FAILURE'])
    datasetColumns: Optional[List] = Field(example=["Age",'Workclass',"Sex"])
    uniqueId: str = Field(example="hex code")

class ExplainOutput(BaseModel):
    status: Status = Field(examples=['SUCCESS', 'FAILURE'])
    payload: Dict = Field(example={})
    api: str = Field(example="https://api.infosys.com/api/v1/explainability/")

#####################################################
# New Mappers for Version-3 related to Explainability
#####################################################
class GetExplanationMethodsRequest(BaseModel):
    modelId: float = Field(example=11.0)
    datasetId: float = Field(example=12.0)
    scope: Optional[Scope ]= Field(example="")
    
    class Config:
        from_attributes = True

class GetExplanationMethodsResponse(BaseModel):
    status: Status = Field(examples=['SUCCESS', 'FAILURE'])
    message: str = Field(example="Identification of explanation methods successful")
    dataType: Optional[str] = Field(example="Tabular")  
    methods: List = Field(example=[])
    
    class Config:
        from_attributes = True
        
class GetExplanationRequest(BaseModel):
    modelId: float = Field(example=11.01)
    datasetId: float = Field(example=12.02)
    preprocessorId: Optional[float] = Field(example=13.03)
    scope: str = Field(example="LOCAL")
    method: str = Field(example="ANCHOR-TABULAR")
    inputText: Optional[str] = Field(example="This movie was fantastic! The plot was gripping and the acting was top-notch.")
    inputRow: Optional[Dict] = Field(example={})
    
    class Config:
        from_attributes = True

class ExplainabilityTabular(BaseModel):
    modelPrediction: Optional[str] = Field(example="Class_GOOD")
    # modelConfidence: Optional[float] = Field(example= "0.85")
    anchorText: Optional[List] = Field(example=["awesome"])
    integratedText: Optional[List] = Field(example=["awesome"])
    anchor: Optional[List] = Field(example=["awesome"])
    importantFeatures : Optional[List] # [importantFeatures]
    limeTimeSeries : Optional[str]  = Field(example="base64 explain image")
    shapValues: Optional[List] = Field(example=[])
    inputRow: Optional[List] = Field(example=[{}])
    # inputText: Optional[str] = Field(example="")
    description: Optional[str] = Field(example="")
    modelDetails: Optional[Dict] = Field(example={})
    datasetDetails: Optional[Dict] = Field(example={})
    
    class Config:
        from_attributes = True
        
class ExplainabilityTabular_1(BaseModel):
    inputText: Optional[str] = Field(example="This movie was fantastic! The plot was gripping and the acting was top-notch.")
    inputRow: Optional[List] = Field(example=[])
    modelPrediction: Optional[str] = Field(example="Normal")
    # modelConfidence: Optional[str] = Field(example="0.85")
    explanation: Optional[List] = Field(example=[])
    timeSeries : Optional[str]  = Field(example="base64 explain image")
    
    class Config:
        from_attributes = True
        
class ExplainabilityTabular_New(BaseModel):
    modelName: str = Field(example="BMI Classification Model")
    algorithm: str = Field(example="Random Forest")
    endpoint: Optional[str] = Field(example="https://localhost:5000/bmi/predict")
    taskType: str = Field(example="CLASSIFICATION")
    datasetName: str = Field(example="BMI Dataset")
    dataType: str = Field(example="Tabular")
    groundTruthLabel: Optional[str] = Field(example="Index")
    groundTruthClassNames: Optional[List] = Field(example=["Underweight", "Normal", "Overweight", "Obese"])
    featureNames: Optional[List] = Field(example=["Age", "Weight", "Height"])
    methodName: str = Field(example="ANCHOR-TABULAR")
    methodDescription: str = Field(example="Anchor Explanation Description")
    anchor: Optional[List[ExplainabilityTabular_1]]
    attributionsText: Optional[List[ExplainabilityTabular_1]]
    featureImportance: Optional[List[ExplainabilityTabular_1]]
    timeSeriesForecast : Optional[List[ExplainabilityTabular_1]]
    shapValues: Optional[List[ExplainabilityTabular_1]]
    explanationDesc: Optional[Dict] = Field(example={})
    
    class Config:
        from_attributes = True

class GetExplanationResponse(BaseModel):
    status: Status = Field(examples=['SUCCESS', 'FAILURE'])
    message: str = Field(example="Identification of explanation methods successful")
    explanation: List[ExplainabilityTabular_New]
    
    class Config:
        from_attributes = True

class GetReportRequest(BaseModel):
    batchId: float = Field(example=123.0)
    
    class Config:
        from_attributes = True

class GetReportResponse(BaseModel):
    status: Status = Field(examples=['SUCCESS', 'FAILURE'])
    message: str = Field(example="Report generated successfully")
    
    class Config:
        from_attributes = True

class ExplainabilityLocalImageResponse(BaseModel):
    explanation: str=Field(example="base64 explain image")
    segments : str=Field(example="base64 segment explain image")
    
    class Config:
        from_attributes = True

class ExplainabilityLocalImageRequest(BaseModel):
    inputImagePath: str = Field(example="responsible-ai//responsible-ai-explain//images//1//inputImage_1.jpg")
    modelPredictionEndpoint: str = Field(example="responsible-ai//responsible-ai-explain//models//2//localImage_explain_model.pkl")
    methods: LocalExplainabilityImageMethods = Field(example="ANCHOR-IMAGE")
    segmentationType: Optional[SegmentationType] = Field(example="slic")
    outputStorageType: str = Field(example="NUTANIX")
    outputType: str = Field(example="JSON")
    outputPath: str = Field(example="responsible-ai//responsible-ai-explain//output//1//localImage_explain.json")

    class Config:
        from_attributes = True

class ExplainabilityLocalTextRequest(BaseModel):
    inputText: str = Field(example="Unfortunately the movie has bad story")
    explainerID: int = Field(example=1)
    taskType: Optional[TaskType] = Field(example="CLASSIFICATION")
    methods: LocalExplainabilityTextMethods = Field(example="ANCHOR-TEXT")
    segmentationType: Optional[SegmentationType] = Field(example="gausslegendre")
    modelPredictionEndpoint: str = Field(example="responsible-ai//responsible-ai-explain//models//2//localtext_explain_model.pkl")
    vectorizerPredictionEndpoint: Optional[str] = Field(
        example="responsible-ai//responsible-ai-explain//models//2//localtext_explain_vectorizer.pkl")
    targetClassNames: List = Field(example=["Negative", "Positive"])
    outputStorageType: str = Field(example="NUTANIX")
    outputType: str = Field(example="JSON")
    outputPath: str = Field(example="responsible-ai//responsible-ai-explain//output//1//localText_explain.json")

    class Config:
        from_attributes = True

class ExplainabilityLocalText(BaseModel):
    predictedTarget: str = Field(example="Class_GOOD")
    anchor: Optional[List] = Field(example=["awesome"])
    attributions: Optional[str] = Field(example="<mark style=background-color:#f8f2f5>string</mark>")

    class Config:
        from_attributes = True

class ExplainabilityLocalTextResponse(BaseModel):
    explainerID: int = Field(example=1)
    explanation: List[ExplainabilityLocalText]
    
    class Config:
        from_attributes = True

class ExplainabilityCoTRequest(BaseModel):
    Prompt: str = Field(example="Which is the biggest country in the world?")
    Response: str = Field(example="The largest country in the world by land area is Russia. It spans over 17 million square kilometers (about 6.6 million square miles), making it significantly larger than any other country. Russia stretches across Eastern Europe and northern Asia, covering 11 time zones and encompassing a wide variety of landscapes and climates.")

    class Config:
        from_attributes = True

class ExplainabilityCoTResponse(BaseModel):
    CoT: Dict = Field(example={})
    Explanation: Dict = Field(example={})
    
    class Config:
        from_attributes = True

        