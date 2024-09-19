"""
 Copyright 2024 Infosys Ltd.”

Use of this source code is governed by MIT license that can be found in the LICENSE file or at
MIT license https://opensource.org/licenses/MIT

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from fairness.dao.bias_model import Bias, TrainingDataset, PredictionDataset
from fairness.dao.mitigation_model import Mitigation, TrainingDataset
from infosys_responsible_ai_fairness.responsible_ai_fairness import BiasResult, DataList, MitigationResult, PRETRAIN, utils, StandardDataset
from infosys_responsible_ai_fairness.responsible_ai_fairness import metricsEntity as me
import openai
import numpy as np
from bson import ObjectId
from fairness.dao.WorkBench.FileStoreDb import FileStoreReportDb
from fairness.dao.individual_fairness import Individual_Fairness
from fairness.dao.llm_connection_credentials import LlmConnectionCredentials
from fairness.dao.llm_analysis import LlmAnalysis
from fairness.dao.model_mitigation_mapper import MitigationModel
from fairness.dao.databaseconnection import DataBase
from io import StringIO, BytesIO
from nutanix_object_storage.nutanix_utility import NutanixObjectStorage
from fastapi.responses import FileResponse
from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate
from fairlearn.postprocessing import ThresholdOptimizer
import joblib
import json
import datetime
import time
import os
# import pdfkit
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import requests
from sklearn.neighbors import NearestNeighbors

from fairness.dao.WorkBench.FileStoreDb import FileStoreReportDb
from fairness.dao.WorkBench.databaseconnection import DataBase_WB
from fairness.mappers.mappers import BiasAnalyzeResponse, BiasAnalyzeRequest, BiasPretrainMitigationResponse, BiasResults, IndividualFairnessRequest, \
    metricsEntity, MitigateBiasRequest, MitigationAnalyzeResponse, PreprocessingMitigationAnalyzeResponse, PreprocessingMitigateBiasRequest, BatchId
from fairness.exception.exception import FairnessException, FairnessUIParameterNotFoundError
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from fairness.dao.WorkBench.Tenet import Tenet
from fairness.dao.WorkBench.Batch import Batch
from fairness.dao.WorkBench.html import Html
from fairness.dao.WorkBench.Data import Dataset, DataAttributes, DataAttributeValues
from fairness.dao.WorkBench.model import Model, ModelAttributes, ModelAttributeValues
from fairness.constants.local_constants import *
from fairness.constants.llm_constants import OPENAI, PROMPT_TEMPLATE, GPT_4
from fairness.config.logger import CustomLogger
from fairness.service.service_utils import Utils
from sklearn.model_selection import train_test_split
import pandas
from fastapi import HTTPException

log = CustomLogger()


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class ModelMitigation:
    def __init__(self, MockDB=None):
        if MockDB is not None:
            self.db = MockDB.db
            self.fileStore = FileStoreReportDb(self.db)
            self.batch =  Batch(self.db)
            self.tenet =  Tenet(self.db)
            self.model = Model(self.db)
            self.modelAttributes = ModelAttributes(self.db)
            self.modelAttributeValues = ModelAttributeValues(self.db)
            self.dataset = Dataset(self.db)
            self.dataAttributes = DataAttributes(self.db)
            self.dataAttributeValues = DataAttributeValues(self.db)
            print("Mockdb is executed")
            self.utils = Utils()
            self.bias_collection = self.db['bias']
            self.mitigation_collection = self.db['mitigation']
            self.mitigation_model_collection = self.db['mitigation_model']
            self.metrics_collection = self.db['metrics']
            self.llm_analysis_collection = self.db['llm_analysis']
            self.llm_connection_credentials_collection = self.db['llm_connection_credentails']

        else:
            self.db = DataBase().db
            self.fileStore = FileStoreReportDb()
            self.batch =  Batch()
            self.model = Model()
            self.modelAttributes = ModelAttributes()
            self.modelAttributeValues = ModelAttributeValues()
            self.tenet =  Tenet()
            self.dataset = Dataset()
            self.dataAttributes = DataAttributes()
            self.dataAttributeValues = DataAttributeValues()
            self.utils = Utils()
            self.bias_collection = self.db['bias']
            self.mitigation_collection = self.db['mitigation']
            self.mitigation_model_collection = self.db['mitigation_model']
            self.metrics_collection = self.db['metrics']
            self.llm_analysis_collection = self.db['llm_analysis']
            self.llm_connection_credentials_collection = self.db['llm_connection_credentails']

    request_payload = ""
    mitigation_payload = ""
    pretrainMitigation_payload = ""
    ca_dict = {}

    def mitigation_modelUpload_Fs(self, payload: dict):
        print(payload)


        train_Id = payload['trainingDatasetID']
        # test_Id = payload['predicitDatasetID']
        model_Id = payload['modelID']
        print("TrainingDataset",train_Id)
        # print("TestDataset",test_Id)
        print("Model",model_Id)
        
        traindata = self.fileStore.read_file(train_Id)
        df_train = pandas.read_csv(BytesIO(traindata['data']))
        print(df_train,"Train_dataset")

        modeldata = self.fileStore.read_modelfile(model_Id)
        model_content = joblib.load(BytesIO(modeldata['data']))

        model_file_name = modeldata['name']
       
        last_dot_index = model_file_name.rfind('.')
        model_file_name = model_file_name[: last_dot_index if last_dot_index != -1 else len(
            model_file_name)]
        model_file_unique_name = model_file_name+'_' + \
            datetime.datetime.now().strftime("%m%d%Y%H%M%S")

        training_dataset_name = traindata['name']
        last_dot_index = training_dataset_name.rfind('.')
        training_dataset_name = training_dataset_name[: last_dot_index if last_dot_index != -1 else len(
            training_dataset_name)]
        training_dataset_unique_name = training_dataset_name + \
            '_'+datetime.datetime.now().strftime("%m%d%Y%H%M%S")

        feature_list = list(df_train.columns)

        response = {
            'feature_list': feature_list
        }

        return response

    

    def mitigation_model_analyze(y_true, y_pred, df_sensitive_features):
        return {
            'demographic_parity_difference': demographic_parity_difference(y_true, y_pred, sensitive_features=df_sensitive_features),
            'equalized_odds_difference': equalized_odds_difference(y_true, y_pred, sensitive_features=df_sensitive_features),
            'true_positive_rate': true_positive_rate(y_true, y_pred),
            'true_negative_rate': true_negative_rate(y_true, y_pred),
            'false_positive_rate': false_positive_rate(y_true, y_pred),
            'false_negative_rate': false_negative_rate(y_true, y_pred),
            'accuracy_score': accuracy_score(y_true, y_pred)
        }

    def mitigation_getmitigated_modelname_analyze(self, payload: dict):
        print(payload,"***************************")
        if payload.Batch_id is None or '':
            log.error("Batch Id id missing")
        batchId = payload.Batch_id
        self.batch.update(batch_id=batchId, value={"Status": "In-progress"})
        tenet_id = self.tenet.find(tenet_name='Fairness')
        batch_details = self.batch.find(batch_id=batchId, tenet_id=tenet_id)
        print(batch_details,"batch_details")
        datasetId = batch_details['DataId']
        # seconddasetId = batch_details['SecondDataId']
        print("datasetId",datasetId)
        modelId = batch_details['ModelId']
        print("modelId",modelId)
        model_details = self.model.find(model_id=modelId)
        modeldata = model_details['ModelData']
        print(modeldata,"modeldata")
        dataset_details = self.dataset.find(Dataset_Id=datasetId)
        # second_dataset_details = Dataset.findsecondfile(secondDataset_Id=seconddasetId)
        sampledata = dataset_details['SampleData']
        print(sampledata,"sampledata")
        # second_dataset = second_dataset_details['SampleData']
        # print(second_dataset,"second_dataset")
        model_details = self.model.find(model_id=modelId)
        print(model_details,"model_details")
        model_attribute_ids = self.modelAttributes.find(model_attributes=[
                                                    'label', 'sensitiveFeatures'])
        print("model Attribute Ids:", model_attribute_ids)
        model_attribute_values = self.modelAttributeValues.find(
            model_id=modelId, model_attribute_ids=model_attribute_ids, batch_id=batchId)

        print(model_attribute_values,"model_attribute values")


        label = model_attribute_values[0]
        print("label", label)
        sensitive_features = model_attribute_values[1]
        print("sensitive_features",sensitive_features)
        traindata = self.fileStore.read_file(sampledata)
        df_train = pandas.read_csv(BytesIO(traindata['data']))
        train, test = train_test_split(df_train, test_size=0.3)

        modeldata = self.fileStore.read_file(modeldata)
        model= joblib.load(BytesIO(modeldata['data']))

        model_file_name = modeldata['name']

        x_train = train.drop(label, axis=1)
        y_train = train[label]

        x_test=test.drop(label, axis=1)
        y_test=test[label]

        df_train_sensitive_features = train[sensitive_features]
        df_test_sensitive_features = test[sensitive_features]

        metrics_before_optimization = {}
        metrics_after_optimization = {}

        y_pred = model.predict(x_test)

        metrics_before_optimization = ModelMitigation.mitigation_model_analyze(
            y_test, y_pred, df_test_sensitive_features)

        optimal_optimizer = None
        optimal_equalized_odds_difference_value = None
        optimal_y_pred = None

        iteration_count = 12

        for iteration in range(iteration_count):
            optimizer = ThresholdOptimizer(estimator=model, constraints='equalized_odds',
                                           objective='balanced_accuracy_score', predict_method='auto', prefit=True)

            optimizer.fit(x_train, y_train,
                          sensitive_features=df_train_sensitive_features)

            y_pred = optimizer.predict(
                x_test, sensitive_features=df_test_sensitive_features)

            current_equalized_odds_difference_value = equalized_odds_difference(
                y_test, y_pred, sensitive_features=df_test_sensitive_features)

            if (optimal_optimizer == None or current_equalized_odds_difference_value < optimal_equalized_odds_difference_value):
                optimal_optimizer = optimizer
                optimal_equalized_odds_difference_value = current_equalized_odds_difference_value
                optimal_y_pred = y_pred

            if (optimal_equalized_odds_difference_value < metrics_before_optimization['equalized_odds_difference']):

                break

        metrics_after_optimization = ModelMitigation.mitigation_model_analyze(
            y_test, optimal_y_pred, df_test_sensitive_features)

        model_unique_name = 'mitigated_model_' + \
            datetime.datetime.now().strftime("%m%d%Y%H%M%S")
        local_path_model = MITIGATED_MODEL_LOCAL_PATH + \
            model_unique_name+'.joblib'

        joblib.dump(optimal_optimizer, local_path_model)

        filePath = local_path_model

        file_id = self.utils.uploadfile_to_mongodb(filePath)
        print(file_id,"file_id")

        able_to_optimize = metrics_after_optimization[
            'equalized_odds_difference'] <= metrics_before_optimization['equalized_odds_difference']
        try:

            response = {
                'mitigated_model_file_name': model_unique_name+'.joblib',
                'able_to_optimize': able_to_optimize,
            }
            self.batch.update(batch_id=batchId, value={"Status": "Completed"})
        except FairnessUIParameterNotFoundError as cie:
                self.batch.update(batch_id=batchId, value={'Status': "Failed"})
                log.error(cie.__dict__)
                log.info("exit JSON ANALYSE method of Fairness Service")
                raise HTTPException(**cie.__dict__)

        if (able_to_optimize):
            response['metrics_before_optimization'] = metrics_before_optimization
            response['metrics_after_optimization'] = metrics_after_optimization

        return response

       