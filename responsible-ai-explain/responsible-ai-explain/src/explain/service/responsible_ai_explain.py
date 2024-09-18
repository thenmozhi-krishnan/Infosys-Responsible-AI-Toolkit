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

import warnings
warnings.filterwarnings('ignore')
import asyncio
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from alibi.explainers import AnchorText, AnchorTabular, PartialDependenceVariance, PermutationImportance, IntegratedGradients, AnchorImage
from alibi.explainers import TreeShap, KernelShap
from alibi.utils import gen_category_map, spacy_model

from aix360.algorithms.tsutils.tsframe import tsFrame
from aix360.algorithms.tslime.tslime import TSLimeExplainer
from aix360.algorithms.tsutils.model_wrappers import Forecaster
from aix360.algorithms.tsutils.tsperturbers import BlockBootstrapPerturber
from lime.lime_tabular import LimeTabularExplainer
from lime import lime_image
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import pad_sequences
from skimage.segmentation import mark_boundaries
from explain.constants.local_constants import *
from explain.config.logger import CustomLogger
from explain.utils.azure import Azure
from explain.utils.prompt_utils import Prompt
import functools
import joblib
import requests
import json
import cv2
import spacy
import shap
import lime

log = CustomLogger()

class ResponsibleAIExplain:

    def split_set(X, y, fraction, random_state=0):
        """
        Given a set X, associated labels y, splits a fraction y from X.
        """
        _, X_split, _, y_split = train_test_split(X,
                                                  y,
                                                  test_size=fraction,
                                                  random_state=random_state,
                                                  )
        log.info(f"Number of records: {X_split.shape[0]}")
        log.info(f"Number of class {0}: {len(y_split) - y_split.sum()}")
        log.info(f"Number of class {1}: {y_split.sum()}")

        return X_split, y_split
    
    def prepare_data(dataset, targetClassLabel, targetClassNames=None):
        """
        Prepares the data and target from the dataset.

        Parameters:
        dataset (DataFrame): The dataset to be prepared.
        targetClassLabel (str): The label of the target class.
        targetClassNames (list, optional): The names of the target classes.

        Returns:
        data, target, featureNames, targetClassNames: The data and target arrays, feature names, and target class names.
        """
        # Check if targetClassLabel is in dataset
        if targetClassLabel in dataset.columns:
            # Prepare the data and target
            data = dataset.drop(targetClassLabel, axis=1, inplace=False)
            target = dataset[targetClassLabel]
        else:
            data = dataset
            target = None

        # Get the feature names
        featureNames = data.columns.to_list()

        # If targetClassNames is not provided, get the unique values from the target
        if targetClassNames is None and target is not None:
            targetClassNames = target.unique()

        # Convert data to numpy array
        data = data.values

        # Convert target to numpy array if it is not None
        if target is not None:
            target = target.values

        return data, target, featureNames, targetClassNames
    
    def pipeline_processing(pipeline, dataset, targetClassLabel, inputData=None):
        """
        Given a pipeline (which includes preprocessing steps and a model) and a dataset, 
        this function preprocesses the dataset, optionally encodes categorical target class labels, 
        and returns the extracted model from the pipeline, transformed dataset, the preprocessed class label and optionally the transformed input data if provided.

        Parameters:
        pipeline : A list where the last element is expected to be a tuple containing the model and its preprocessor.
        dataset (DataFrame): The dataset to be preprocessed, expected to be a pandas DataFrame.
        targetClassLabel (str): The name of the column in the preprcessed dataset that contains the target class labels.
        inputData (DataFrame, optional): The input data to be transformed using the preprocessor.

        Returns:
        model, transformed_dataset, preprocessed_class_label, transformed_input_data : The extracted model from the pipeline, the dataset after applying preprocessing, the preprocessed class label, and optionally the transformed input data if provided.
        """
        # Initialize variables
        transformed_data = None
        preprocessed_class_label = None
        preprocessor_features_count = 0

        # Extract the model and preprocessor from the pipeline
        model = pipeline.steps[-1][1]
        preprocessor = pipeline[:-1]
        dataset_columns_count = len(dataset.columns)

        # Check if the preprocessor has a feature count attribute
        if hasattr(preprocessor, 'n_features_in_'):
            preprocessor_features_count = preprocessor.n_features_in_

        if inputData is not None:
            inputData = pd.DataFrame(preprocessor.transform(inputData), columns=preprocessor.get_feature_names_out())

        #  Check dataset columns count and preprocessor features count
        if dataset_columns_count-1 == preprocessor_features_count:
            preprocessed_class_label = targetClassLabel
            # Extract the features from the dataset except the target class label
            features = dataset.drop(targetClassLabel, axis=1)
            # Transform the features using the preprocessor
            transformed_data= pd.DataFrame(preprocessor.transform(features))
            
            categorical_features = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
            # Check if the target class label is categorical
            if targetClassLabel in categorical_features:
                # Encode the target class labels
                ordinal_encoder = preprocessing.OrdinalEncoder()
                transformed_data[targetClassLabel] = ordinal_encoder.fit_transform(dataset[[targetClassLabel]]).astype(int)
            else:
                # Copy the target class labels as is
                transformed_data[targetClassLabel] = dataset[targetClassLabel]
        # Check if the dataset columns count is equal to the preprocessor features count
        elif dataset_columns_count == preprocessor_features_count:
            # Transform the dataset using the preprocessor
            transformed_data = pd.DataFrame(preprocessor.transform(dataset))
            # Check if the target class label is present in the transformed data
            for i in transformed_data.columns:
                if targetClassLabel in str(i):
                    preprocessed_class_label= str(i)
                    break

        preprocessed_feature_names = preprocessor.get_feature_names_out()
        # Check if the dataset columns count is not equal to the preprocessor features count
        if dataset_columns_count != len(preprocessed_feature_names):
            # Append the target class label to the preprocessed feature names
            preprocessed_feature_names = np.append(preprocessed_feature_names, targetClassLabel)
        # Create a DataFrame from the transformed data
        preprocessed_data = pd.DataFrame(transformed_data.values, columns=preprocessed_feature_names)

        # Return the model, preprocessed data, and preprocessed class label
        return model, preprocessed_data, preprocessed_class_label, inputData
    
    def find_date_column(dataset):
        for column in dataset.columns:
            try:
                # Attempt to convert the column to datetime
                temp_series = pd.to_datetime(dataset[column], errors='coerce')
                # Check if most of the column could be converted successfully
                if not temp_series.isna().mean() > 0.5:  # Arbitrary threshold, adjust based on your data
                    return column
            except Exception as e:
                # If any error occurs during conversion, move to the next column
                continue
        # If no date column is found, raise an exception
        raise ValueError("No date/datetime format column found in the dataset.")


    def hlstr(string, color='white', size='18px', font='sans-serif'):
        """
        Return HTML markup highlighting text with the desired color.
        """
        return f"<mark style='background-color:{color}; font-size:{size}; font-family:{font};'>{string} </mark>"
    
    def colorize(attrs, cmap='PiYG'):
        """
        Compute hex colors based on the attributions for a single instance.
        Uses a diverging colorscale by default and normalizes and scales
        the colormap so that colors are consistent with the attributions.
        """
        import matplotlib as mpl
        cmap_bound = np.abs(attrs).max()
        norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
        cmap = mpl.cm.get_cmap(cmap)

        # now compute hex values of colors
        colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))
        return colors
    
    def endpoint_check(x):
        if x.shape[0]==1:
            response = requests.post(url, json={input_request: x.reshape(1, -1).tolist()})
        else:
            response = requests.post(url, json={input_request: x.tolist()})
        prediction = json.loads(response.text)[output_response]
        
        return np.array(prediction)

    def local_explain_demo(text: str,class_names):
        log.info("Running local_explain")
        try:
            try:
                explanation = Azure().human_readable_explanation(Prompt.get_classification_prompt(text))
                explanation = json.loads(explanation)
            except Exception as e:
                local_explain_text_demo_model = joblib.load('../models/demo/local_explain_text/localtext_lr_model.pkl')
                local_explain_text_demo_vectorizer = joblib.load('../models/demo/local_explain_text/localtext_vectorizer.pkl')
                local_explain_text_demo_predict_fn = lambda x: local_explain_text_demo_model.predict(local_explain_text_demo_vectorizer.transform(x))
                local_explainer_text_demo = AnchorText.load('../models//alibi//', predictor=local_explain_text_demo_predict_fn)
                prediction = class_names[local_explain_text_demo_predict_fn([text])[0]]
            
                explaination = local_explainer_text_demo.explain(text,
                                                threshold=0.95,
                                                stop_on_first=True,
                                                min_samples_start=100,
                                                coverage_samples=100,
                                                tau=0.5
                                                )
                return {"predictedTarget": prediction, "anchor": explaination.anchor, "explanation": None}
            
            return {"predictedTarget": explanation['Sentiment'], "anchor": explanation['Keywords'], "explanation": explanation['Explanation']}

        except Exception as e:
            log.error(e,exc_info=True)
            raise

    def anchor_text(params: dict):
        log.info("Running Local Explanation for Unstructured Text Data using Anchor Text")

        try:
            text = params['text']
            model = params['model']
            vectorizer = params['vectorizer']
            class_names = params['class_names']
            
            if vectorizer is not None:
                predict_fn = lambda x: model.predict(vectorizer.transform(x))
            else:
                predict_fn = lambda x: model.predict(x)

            prediction = class_names[predict_fn([text])[0]]

            explainer = AnchorText.load('../models//alibi//', predictor=predict_fn)
            explaination = explainer.explain(text,
                                             threshold=0.95,
                                             stop_on_first=True,
                                             min_samples_start=100,
                                             coverage_samples=100,
                                             verbose=True
                                             )

            return [{"predictedTarget": prediction, "anchor": explaination.anchor}]

        except Exception as e:
            log.error(e,exc_info=True)
            raise Exception
    def integrated_gradients_text(params: dict):
        log.info("Running Local Explanation for Unstructured Text Data using Integrated Gradients")

        try:
            text = params['text']
            model = params['model']
            vectorizer = params['vectorizer']
            segmentationType = params['segmentationType']
            class_names = params['class_names']
            
            if vectorizer is not None:
                tokenized_sentence = vectorizer.texts_to_sequences([text])
                preprocessed_sentence = pad_sequences(tokenized_sentence, padding='post', maxlen=100)
            else:
                preprocessed_sentence = text

            predictions = ((model(preprocessed_sentence).numpy() > 0.5).astype('int32'))[0] # model(preprocessed_sentence).numpy().argmax(axis=1)
            
            layer = model.get_layer('embedding')
            explainer  = IntegratedGradients(model, 
                                             layer=layer, 
                                             n_steps=50, 
                                             method=segmentationType, 
                                             internal_batch_size=100)
            explanation = explainer.explain(preprocessed_sentence, 
                                            baselines=None, 
                                            target=predictions, 
                                            attribute_to_layer_inputs=False)
            # Get attributions values from the explanation object
            attrs = explanation.attributions[0]
            attrs = attrs.sum(axis=2)

            prediction = class_names[predictions[0]]
            
            words = text.split(" ")
            colors = ResponsibleAIExplain.colorize(attrs[0])

            return [{"predictedTarget": prediction, "attributions": "".join(list(map(ResponsibleAIExplain.hlstr, words, colors)))}]

        except Exception as e:
            log.error(e, exc_info=True)
            raise Exception

    def local_explain_text(text: str, model, vectorizer, method, segmentationType, class_names):

        try:
            params = {
                "text":text,
                "model":model,
                "vectorizer":vectorizer,
                "segmentationType":segmentationType,
                "class_names": class_names
                }

            log.debug(f"method: {method}")

            dict_local_text_explain_mapping = {
                "ANCHOR-TEXT": ResponsibleAIExplain.anchor_text,
                "INTEGRATED-GRADIENTS": ResponsibleAIExplain.integrated_gradients_text
            }

            return dict_local_text_explain_mapping[method](params)

        except Exception as e:
            log.error(e, exc_info=True)
            raise Exception
        
    def local_explain_AnchorImage(params: dict):
        log.info("Running Local Explanation for Image using Anchor Image")
        try:
            model=params['model']
            inputImage=params['inputImage']
            segmentation_fn=params['segmentation_fn']

            img_shape = model.input_shape
            image_shape = (img_shape[1], img_shape[2], img_shape[3])
            resizedImage = cv2.resize(inputImage, (img_shape[1], img_shape[2]))

            predict_fn = lambda x: model.predict(x)

            explainer = AnchorImage(predict_fn, image_shape, segmentation_fn=segmentation_fn,
                                    images_background=None)
            explanation = explainer.explain(resizedImage, threshold=.95, p_sample=.35, tau=0.5)
            
            return {"segments":explanation.segments,"anchor":explanation.anchor}

        except Exception as e:
            log.error(e, exc_info=True)
            raise Exception
        
    def local_explain_IntegratedGradientsImage(params: dict):
        log.info("Running Local Explanation for Image using Integrated Gradients")
        try:
            model=params['model']
            inputImage=params['inputImage']
            segmentation_fn=params['segmentation_fn']
            
            img_shape = model.input_shape
            resizedImage = cv2.resize(inputImage, (img_shape[1], img_shape[2]))
            instance = np.expand_dims(resizedImage, axis=0)
            
            predictions = model(instance).numpy().argmax(axis=1)
            
            explainer = IntegratedGradients(model, method=segmentation_fn)
            explanation = explainer.explain(instance, target=predictions)

            return {"attributions":explanation.attributions[0].squeeze() }

        except Exception as e:
            log.error(e, exc_info=True)
            raise Exception
    
    def local_explain_LimeImage(params: dict):
        log.info("Running Local Explanation for Image using Lime")
        try:
            model=params['model']
            inputImage=params['inputImage']

            img_shape = model.input_shape
            resizedImage = cv2.resize(inputImage, (img_shape[1], img_shape[2]))
            instance = np.expand_dims(resizedImage, axis=0)

            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(instance[0].astype('double'), model.predict, top_labels=5, hide_color=0, num_samples=500, batch_size=32)
            # explanation = explainer.explain_instance(instance[0].astype('double'), model.predict, top_labels=5, hide_color=0, num_samples=500)
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
            lime_exp= mark_boundaries(temp/ 255.0, mask, color=(0, 0, 1), mode='subpixel') # {'thick', 'inner', 'outer', 'subpixel'}
            
            return {"lime_image":lime_exp}

        except Exception as e:
            log.error(e, exc_info=True)
            raise Exception
        
    def local_image_explain(inputImage,
                      model,
                      method,
                      segmentation_fn):
        try:
            params = {
                "inputImage":inputImage,
                "model": model,
                "segmentation_fn":segmentation_fn}

            log.debug(f"method: {method}")

            dict_localImage_explain_mapping = {
                "ANCHOR-IMAGE": ResponsibleAIExplain.local_explain_AnchorImage,
                "INTEGRATED-GRADIENTS": ResponsibleAIExplain.local_explain_IntegratedGradientsImage,  
                "LIME": ResponsibleAIExplain.local_explain_LimeImage 
            }

            return dict_localImage_explain_mapping[method](params)

        except Exception as e:
            log.error(e, exc_info=True)
            raise Exception
    
    def anchor_tabular(params: dict):
        log.info("Running Local Explanation for Structured Tabular Data using Anchor Tabular")
        try:
            dataset = params['dataset']
            inputIndex = params['inputIndex']
            model = params['model']
            preprocessor = params['preprocessor']
            featureNames = params['featureNames']
            categoricalFeatureNames = params['categoricalFeatureNames']
            targetNames = params['targetNames']
            targetClassNames = params['targetClassNames']

            log.debug(f"preprocessor: {preprocessor}")
            if preprocessor is not None:
                predict_fn = lambda x: model.predict(preprocessor.transform(x))
            else:
                predict_fn = lambda x: model.predict(x)

            category_map = None
            if categoricalFeatureNames is not None:
                category_map = gen_category_map(dataset,categorical_columns=categoricalFeatureNames)
                log.debug(f"category_map: {category_map}")

            if preprocessor is not None:
                label_encoder = preprocessing.LabelEncoder()
                target = label_encoder.fit_transform(dataset[targetNames])
            else:
                target = dataset[targetNames]

            if targetClassNames is None:
                targetClassNames = dataset[targetNames].unique()

            X = dataset.drop(targetNames, axis=1, inplace=False)
            log.debug(f"X: {X}")
            if featureNames is not None:
                X = X[featureNames]
            else:
                featureNames = list(X.columns.values)

            if preprocessor is not None:
                label_encoder = preprocessing.LabelEncoder()
                column_names = list(X.columns.values)
                for col in column_names:
                    X[col] = label_encoder.fit_transform(X[col])

                imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                imputer = imputer.fit(X)
                data = imputer.transform(X)
            else:
                data = X

            data_perm = np.random.permutation(np.c_[data, target])
            data = data_perm[:, :-1]
            target = data_perm[:, -1]
            log.debug(f"data: {data}")
            log.debug(f"target: {target}")

            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, train_size=0.80,
                                                                random_state=0)
            explainer = AnchorTabular(
                predictor=predict_fn,
                feature_names=featureNames,
                categorical_names=category_map)
            
            if preprocessor is not None:
                y_pred= model.predict(preprocessor.transform(X_test))
            else:
                y_pred= model.predict(X_test)
            accuracy=accuracy_score(y_test,y_pred)

            explainer.fit(X_train, disc_perc=[25, 50, 75])

            explaination = explainer.explain(X_test[inputIndex], threshold=0.95)
            inputRow=[]
            predict_input = X_test[inputIndex].reshape(1, -1)
            df = pd.DataFrame(data=predict_input, columns=featureNames)
            input = df.to_dict(orient='records')
            for key, val in input[0].items():
                    inputRow.append({"featureName": key, "featureValue": val})
            
            prediction = targetClassNames[int(explainer.predictor(predict_input)[0])]
            return [{"predictedTarget": str(prediction),
                     "anchor": explaination.anchor,
                     "modelConfidence":accuracy,
                     "inputRow":inputRow}]

        except Exception as e:
            log.error(e, exc_info=True)
            return {"predictedTarget": "", "anchor": ""}

    def kernel_shap_local_tabular(params: dict):
        log.info("Running Local Explanation for Structured Tabular Data using Kernel Shap")
        try:
            categoricalFeatureNames = params["categoricalFeatureNames"]
            taskType = params['taskType']
            model = params['model']
            preprocessor = params['preprocessor']
            dataset = params['dataset']
            targetNames = params['targetNames']
            targetClassNames = params['targetClassNames']
            featureNames = params['featureNames']

            if targetClassNames is None:
                targetClassNames = list(dataset[targetNames].unique())

            category_map = None
            if categoricalFeatureNames is not None:
                category_map = gen_category_map(data=dataset, categorical_columns=categoricalFeatureNames)

            X = dataset.drop(targetNames, axis=1, inplace=False)
            
            if featureNames is None:
                featureNames = list(X.columns.values)
            else:
                X = X[featureNames]

            if preprocessor is not None:
                label_encoder = preprocessing.LabelEncoder()
                for col in featureNames:
                    X[col] = label_encoder.fit_transform(X[col])

                imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                imputer = imputer.fit(X)
                data = imputer.transform(X)

                label_encoder = preprocessing.LabelEncoder()
                target = label_encoder.fit_transform(dataset[targetNames])
                
                pred_fcn = lambda x: model.predict(preprocessor.transform(x))
            else:
                data = X
                target = dataset[targetNames]
                
                pred_fcn = lambda x: model.predict(x)

            data_perm = np.random.permutation(np.c_[data, target])
            data = data_perm[:, :-1]
            target = data_perm[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, train_size=0.80,
                                                                random_state=0)

            if taskType=="REGRESSION":
                lr_explainer = KernelShap(pred_fcn, task='regression', feature_names=featureNames)
                y_pred= model.predict(X_test)
                modelConfidence=r2_score(y_test,y_pred)
                
            else:
                lr_explainer = KernelShap(pred_fcn, feature_names=featureNames)
                y_pred= model.predict(X_test)
                modelConfidence=accuracy_score(y_test,y_pred)

            lr_explainer.fit(X_train[0:100])

            predict_input = X_train[params['inputIndex']].reshape(1, -1)

            explanation = lr_explainer.explain(predict_input)

            log.debug(f"explanation: {explanation}")

            log.debug(f"targetClass: {targetClassNames}")
            
            importances = explanation['data']['raw']['importances']
            
            list_kernel_shap = []
            for item in importances.items():
                log.debug(f"item: {item}")
                if taskType == "REGRESSION":
                    if item[0] == 'aggregated':
                        continue
                else:                    
                    if item[0] != str(explanation['raw']['prediction'][0]):
                        continue
                data = dict(zip(item[1]['names'], item[1]['ranked_effect']))
                sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))


                list_importance = []
                inputRow=[]
                df = pd.DataFrame(data=predict_input, columns=featureNames)
                input = df.to_dict(orient='records')
                for key, val in input[0].items():
                    inputRow.append({"featureName": key, "featureValue": val})

                original_list=list(sorted_data.values())
                min_val = min(original_list)
                max_val = max(original_list)
                normalized_list = [1 + 99 * (x - min_val) / (max_val - min_val) for x in original_list]

                sum_normalized = sum(normalized_list)
                scaling_factor = 100 / sum_normalized
                new_normalized_list = [x * scaling_factor for x in normalized_list]
                i=0
                for k in sorted_data.keys():
                    list_importance.append({"name": k, "value": round(new_normalized_list[i],4)})
                    i=i+1

                if taskType == 'REGRESSION':
                    list_kernel_shap.append({"predictedTarget": str(explanation['raw']['raw_prediction'][0]),
                                         "modelConfidence": abs(modelConfidence),
                                         "importantFeatures": list_importance,
                                         "inputRow": inputRow})


                else:
                    list_kernel_shap.append({"predictedTarget": str(targetClassNames[explanation['raw']['prediction'][0]]),
                                            "modelConfidence": modelConfidence,
                                            "importantFeatures": list_importance,
                                             "inputRow": inputRow})

            return list_kernel_shap
        
        except Exception as e:
            log.error(e, exc_info=True)
            raise Exception
        
    def tree_shap_local_tabular(params: dict):
        log.info("Running Local Explanation for Structured Tabular Data using Tree Shap")
        try:
            categoricalFeatureNames = params["categoricalFeatureNames"]
            taskType = params['taskType']
            model = params['model']
            preprocessor = params['preprocessor']
            dataset = params['dataset']
            targetNames = params['targetNames']
            targetClassNames = params['targetClassNames']
            featureNames = params['featureNames']

            if targetClassNames is None:
                targetClassNames = list(dataset[targetNames].unique())

            category_map = None
            if categoricalFeatureNames is not None:
                category_map = gen_category_map(data=dataset, categorical_columns=categoricalFeatureNames)

            X = dataset.drop(targetNames, axis=1, inplace=False)
            
            if featureNames is None:
                featureNames = list(X.columns.values)
            else:
                X = X[featureNames]

            if preprocessor is not None:
                label_encoder = preprocessing.LabelEncoder()
                for col in featureNames:
                    X[col] = label_encoder.fit_transform(X[col])

                imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                imputer = imputer.fit(X)
                data = imputer.transform(X)

                label_encoder = preprocessing.LabelEncoder()
                target = label_encoder.fit_transform(dataset[targetNames])
            else:
                data = X
                target = dataset[targetNames]

            data_perm = np.random.permutation(np.c_[data, target])
            data = data_perm[:, :-1]
            target = data_perm[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, train_size=0.80,
                                                                random_state=0)

            if taskType == "REGRESSION":
                tree_explainer_interventional = TreeShap(model, 
                                                    task='regression',
                                                    feature_names=featureNames, 
                                                    categorical_names=category_map)
                y_pred= model.predict(X_test)
                modelConfidence=r2_score(y_test,y_pred)

            else:
                tree_explainer_interventional = TreeShap(model, 
                                                        feature_names=featureNames, 
                                                        categorical_names=category_map)
                y_pred= model.predict(X_test)
                modelConfidence=accuracy_score(y_test,y_pred)
            tree_explainer_interventional.fit()

            predict_input = X_train[params['inputIndex']].reshape(1, -1)

            explanation = tree_explainer_interventional.explain(predict_input,
                                                                check_additivity=False)

            log.debug(f"explanation: {explanation}")
            log.debug(f"targetClass: {targetClassNames}")
            
            importances = explanation['data']['raw']['importances']
            
            list_kernel_shap = []
            for item in importances.items():
                log.debug(f"item: {item}")
                if taskType == "REGRESSION":
                    if item[0] == 'aggregated':
                        continue
                else:                    
                    if item[0] != str(explanation['raw']['prediction'][0]):
                        continue
                data = dict(zip(item[1]['names'], item[1]['ranked_effect']))
                sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))


                list_importance = []
                inputRow=[]
                df = pd.DataFrame(data=predict_input, columns=featureNames)
                input = df.to_dict(orient='records')
                for key, val in input[0].items():
                    inputRow.append({"featureName": key, "featureValue": val})

                original_list=list(sorted_data.values())
                min_val = min(original_list)
                max_val = max(original_list)
                normalized_list = [1 + 99 * (x - min_val) / (max_val - min_val) for x in original_list]

                sum_normalized = sum(normalized_list)
                scaling_factor = 100 / sum_normalized
                new_normalized_list = [x * scaling_factor for x in normalized_list]
                i=0
                for k in sorted_data.keys():
                    list_importance.append({"name": k, "value": round(new_normalized_list[i],4)})
                    i=i+1

                if taskType == "REGRESSION":
                    list_kernel_shap.append({"predictedTarget": str(explanation['raw']['raw_prediction']),
                                            "modelConfidence": abs(modelConfidence),
                                            "importantFeatures": list_importance,
                                             "inputRow":inputRow})

                else:
                    list_kernel_shap.append({"predictedTarget": str(targetClassNames[explanation['raw']['prediction'][0]]),
                                         "modelConfidence": modelConfidence,
                                         "importantFeatures": list_importance,
                                         "inputRow":inputRow})


            return list_kernel_shap
        
        except Exception as e:
            log.error(e, exc_info=True)
            raise Exception
    
    def timeSeries(params: dict):
        log.info("Running TimeSeries Local Explanation for Structured Tabular Data")
        try:
            taskType = params['taskType']
            model = params['model']
            preprocessor = params['preprocessor']
            dataset = params['dataset']
            targetNames = params['targetNames']
            inputIndex = params['inputIndex']
            featureNames = params['featureNames']
            
            X = dataset.drop(targetNames, axis=1, inplace=False)

            if featureNames is None:
                featureNames = list(X.columns.values)
            else:
                X = X[featureNames]
            
            if preprocessor is not None:
                label_encoder = preprocessing.LabelEncoder()
                for col in featureNames:
                    X[col] = label_encoder.fit_transform(X[col])

                imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                imputer = imputer.fit(X)
                data = imputer.transform(X)

                label_encoder = preprocessing.LabelEncoder()
                target = label_encoder.fit_transform(dataset[targetNames])
            else:
                data = X
                target = dataset[targetNames]

            data_perm = np.random.permutation(np.c_[data, target])
            data = data_perm[:, :-1]
            target = data_perm[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, train_size=0.80,
                                                                random_state=0)
            f_model = Forecaster(model=model, forecast_function="predict")
            relevant_history = 25
            explainer = TSLimeExplainer(
                                        model= functools.partial(f_model.predict), # , verbose = 0
                                        input_length=120,
                                        relevant_history=relevant_history,
                                        perturbers=[
                                            BlockBootstrapPerturber(window_length=min(10, 120-1), block_length=2, block_swap=2),
                                        ],
                                        n_perturbations=10000,
                                        random_seed=22,
                                    )

            sequence_length = 120
            instance = X_test[inputIndex:inputIndex+sequence_length]
            
            ts_instance = tsFrame(instance)
            ts_instance.index = pd.to_numeric(ts_instance.index)
            explanation = explainer.explain_instance(ts_instance)

            selected_feature_columns = featureNames
            selected_feature_names = featureNames

            fig = plt.figure(layout='constrained', figsize=(15, 10))
            gs0 = gridspec.GridSpec(2, 1, figure=fig, height_ratios= [1, 3], top=0.9)
            gs2 = gridspec.GridSpecFromSubplotSpec(7, 3, subplot_spec=gs0[1])
            gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[0], width_ratios=[4.5, 10, 4.5])

                    
            def plot_feature(figure, scores, feature_values, gs, title, legend=False):
                ax = fig.add_subplot(gs)
                ax.plot(feature_values, label='Input Time Series', marker='o')
                ax.set_title(title)

                ax.bar(feature_values.index, scores, 0.4, label = 'TSLime Weights (Normalized)', color='red')
                ax.axhline(y=0, color='r', linestyle='-', alpha=0.4)
                if legend:
                    ax.legend(bbox_to_anchor=(1.5, 1.0), loc='upper right')

            instance_prediction = explanation['model_prediction'][0]
            normalized_weights = (explanation['history_weights'] / np.mean(np.abs(explanation['history_weights'])))
            modelConfidence = r2_score(y_test[inputIndex:inputIndex+sequence_length], explanation['model_prediction'])

            for i, feature_col in enumerate(selected_feature_columns):
                if i > 0:
                    gs = gs2[i-1]
                else:
                    gs = gs1[1]
                
                plot_feature(figure=fig,
                            scores=normalized_weights[:,i].flatten(), 
                            feature_values=ts_instance.iloc[-relevant_history:, i],
                            gs=gs,
                            title="{} - {}".format(selected_feature_names[i], feature_col),
                            legend=(i==0))

            fig.suptitle("Time Series Lime Explanation Plot with forecast={}".format(str(instance_prediction)), fontsize="x-large")
            # plt.show()
            l1 = []
            l1.append(fig)
            list_time_series=[]
                
            list_time_series.append({"predictedTarget": str(instance_prediction),
                                         "modelConfidence": modelConfidence,
                                         "limeTimeSeries": l1})

            return list_time_series
        
        except Exception as e:
            log.error(e, exc_info=True)
            raise Exception

    def local_explain_tabular(dataset,
                       inputIndex,
                       taskType,
                       method,
                       model,
                       preprocessor,
                       featureNames,
                       categoricalFeatureNames,
                       targetNames,
                       targetClassNames):
        try:
            
            params = {
                "dataset":dataset,
                "inputIndex": inputIndex,
                "taskType": taskType,
                "model":model,
                "preprocessor":preprocessor,
                "featureNames": featureNames,
                "categoricalFeatureNames":categoricalFeatureNames,
                "targetNames":targetNames,
                "targetClassNames":targetClassNames}

            log.debug(f"method: {method}")

            dict_local_explain_mapping = {
                "ANCHOR-TABULAR": ResponsibleAIExplain.anchor_tabular,
                "KERNEL-SHAP": ResponsibleAIExplain.kernel_shap_local_tabular,
                "TREE-SHAP": ResponsibleAIExplain.tree_shap_local_tabular,
                "TS-LIMEEXPLAINER" :ResponsibleAIExplain.timeSeries,
            }

            return dict_local_explain_mapping[method](params)

        except Exception as e:
            log.error(e,exc_info=True)
            raise Exception

    def partial_dependence_variance(params: dict):
        log.info("Running Global Explanation for Structured Tabular Data using Partial Dependence Variance")
        try:
            dataset = params['dataset']
            taskType = params['taskType']
            targetNames = params['targetNames']
            model = params['model']
            preprocessor = params['preprocessor']
            categoricalFeatureNames = params["categoricalFeatureNames"]
            targetClassNames = params['targetClassNames']

            if targetClassNames is None:
                targetClassNames = list(dataset[targetNames].unique())

            category_map = None
            if categoricalFeatureNames is not None:
                category_map = gen_category_map(data=dataset, categorical_columns=categoricalFeatureNames)
            
            X = dataset.drop(targetNames, axis=1, inplace=False)

            column_names = list(X.columns.values)
            if params['featureNames'] is None:
                feature_names = column_names
            else:
                feature_names = params['featureNames']
                X = X[feature_names]
                column_names = list(X.columns.values)


            label_encoder = preprocessing.LabelEncoder()
            for col in column_names:
                X[col] = label_encoder.fit_transform(X[col])

            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer = imputer.fit(X)
            X = imputer.transform(X)

            pd_variance = PartialDependenceVariance(predictor=model
                                                    ,target_names=targetClassNames
                                                    ,categorical_names=category_map
                                                    ,feature_names=feature_names)
            exp_importance = pd_variance.explain(X=X,
                                                method='importance'
                                                )
            log.debug(f"exp_importance: {exp_importance}")
            
            data = dict(zip(column_names, exp_importance.feature_importance[0]))
            log.debug(f"data: {data}")
            
            sorted_data = dict(sorted(data.items(), key = lambda x: x[1], reverse = True))
            log.debug(f"sorted_data: {sorted_data}")

            list_data = []
            original_list=list(sorted_data.values())
            min_val = min(original_list)
            max_val = max(original_list)
            normalized_list = [1 + 99 * (x - min_val) / (max_val - min_val) for x in original_list]
            sum_normalized = sum(normalized_list)
            scaling_factor = 100 / sum_normalized
            new_normalized_list = [x * scaling_factor for x in normalized_list]
            i=0
            for k in sorted_data.keys():
                list_data.append({"featureName": k, "importanceScore": round(new_normalized_list[i],4)})
                i=i+1

            return [{"importantFeatures":list_data}]

        except Exception as e:
            log.error(e,exc_info=True)
            raise Exception
        
    def permutation_importance(params: dict):
        log.info("Running Global Explanation for Structured Tabular Data using Permutation Importance")
        try:
            dataset = params['dataset']
            model = params['model']
            taskType = params['taskType']
            preprocessor = params['preprocessor']
            featureNames = params["featureNames"]
            categoricalFeatureNames = params['categoricalFeatureNames']
            targetNames = params['targetNames']
            targetClassNames = params['targetClassNames']

            if targetClassNames is None:
                targetClassNames = list(dataset[targetNames].unique())
                
            if preprocessor is None:
                prediction_fn = lambda x: np.round(model.predict(x))
            else:
                prediction_fn = lambda x: np.round(model.predict(preprocessor.transform(x)))

            y = dataset[targetNames].to_numpy()
            X = dataset.drop(targetNames, axis=1, inplace=False)

            if featureNames is None:
                featureNames = list(X.columns.values)

            X = X[featureNames].to_numpy()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.80,
                                                                random_state=0)
            if taskType=="REGRESSION":
                pi_explainer = PermutationImportance(predictor=prediction_fn, 
                                                 loss_fns=['mean_absolute_error'], 
                                                 feature_names=featureNames)
            else:
                pi_explainer = PermutationImportance(predictor=prediction_fn, 
                                                    score_fns=['accuracy', 'f1'], 
                                                    feature_names=featureNames)
            exp_importance = pi_explainer.explain(X=X_test,
                                                 y=y_test)
            
            log.debug(f"exp_importance: {exp_importance}")
            
            l = []
            for i in range(len(exp_importance.feature_importance[0])):
                l.append(exp_importance.feature_importance[0][i]['mean'])
            data = dict(zip(featureNames, np.array(l)))
            log.debug(f"data: {data}")
            
            sorted_data = dict(sorted(data.items(), key = lambda x: x[1], reverse = True))
            log.debug(f"sorted_data: {sorted_data}")

            list_data = []
            original_list=list(sorted_data.values())
            min_val = min(original_list)
            max_val = max(original_list)
            normalized_list = [1 + 99 * (x - min_val) / (max_val - min_val) for x in original_list]
            sum_normalized = sum(normalized_list)
            scaling_factor = 100 / sum_normalized
            new_normalized_list = [x * scaling_factor for x in normalized_list]
            i=0
            for k in sorted_data.keys():
                list_data.append({"featureName": k, "importanceScore": round(new_normalized_list[i],4)})
                i=i+1
            return [{"importantFeatures":list_data}]

        except Exception as e:
            log.error(e, exc_info=True)
            raise Exception

    def kernel_shap(params: dict):
        log.info("Running Global Explanation for Structured Tabular Data using Kernel Shap")
        try:
            categoricalFeatureNames = params["categoricalFeatureNames"]
            classifier = params['model']
            preprocessor = params['preprocessor']
            dataset = params['dataset']
            targetNames = params['targetNames']
            targetClassNames = params['targetClassNames']

            if targetClassNames is None:
                targetClassNames = list(dataset[targetNames].unique())

            category_map = None
            if categoricalFeatureNames is not None:
                category_map = gen_category_map(data=dataset, categorical_columns=categoricalFeatureNames)

            X = dataset.drop(targetNames, axis=1, inplace=False)

            column_names = list(X.columns.values)
            if params['featureNames'] is None:
                feature_names = column_names
            else:
                feature_names = params['featureNames']
                X = X[feature_names]
                column_names = list(X.columns.values)

            if preprocessor is not None:
                label_encoder = preprocessing.LabelEncoder()
                for col in column_names:
                    X[col] = label_encoder.fit_transform(X[col])

                imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                imputer = imputer.fit(X)
                data = imputer.transform(X)

                label_encoder = preprocessing.LabelEncoder()
                target = label_encoder.fit_transform(dataset[targetNames])
            else:
                data = X
                target = dataset[targetNames]

            data_perm = np.random.permutation(np.c_[data, target])
            data = data_perm[:, :-1]
            target = data_perm[:, -1]


            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, train_size=0.80,
                                                                random_state=0)
            fraction_explained = 0.05
            X_explain, y_explain = ResponsibleAIExplain.split_set(X_test,
                                                                  y_test,
                                                                  fraction_explained,
                                                                  )

            if preprocessor is not None:
                X_train_proc = preprocessor.transform(X_train)
                X_test_proc = preprocessor.transform(X_test)
                X_explain_proc = preprocessor.transform(X_explain)

                ohe = preprocessor.transformers_[1][1].named_steps['onehot']

                feat_enc_dim = [len(cat_enc) - 1 for cat_enc in ohe.categories_]

                numerical_feats_idx = preprocessor.transformers_[0][2]
                categorical_feats_idx = preprocessor.transformers_[1][2]

                num_feats_names = [feature_names[i] for i in numerical_feats_idx]
                cat_feats_names = [feature_names[i] for i in categorical_feats_idx]

            else:
                X_train_proc = X_train
                X_test_proc = X_test
                X_explain_proc = X_explain
                num_feats_names = feature_names
                cat_feats_names = []

            perm_feat_names = num_feats_names + cat_feats_names

            start_example_idx = 0
            stop_example_idx = 100
            background_data = slice(start_example_idx, stop_example_idx)

            pred_fcn = lambda x: classifier.predict_proba(x)

            lr_explainer = KernelShap(pred_fcn, link='logit', feature_names=perm_feat_names)
            lr_explainer.fit(X_train_proc[background_data, :])

            if category_map is not None:
                ordinal_features = [x for x in range(len(feature_names)) if x not in list(category_map.keys())]
                start = len(ordinal_features)
                cat_feat_start = [start]
                for dim in feat_enc_dim[:-1]:
                    cat_feat_start.append(dim + cat_feat_start[-1])
            else:
                cat_feat_start = None
                feat_enc_dim = None

            explanation = lr_explainer.explain(X_explain_proc[0:1],
                                               summarise_result=True,
                                               cat_vars_start_idx=cat_feat_start,
                                               cat_vars_enc_dim=feat_enc_dim
                                               )
            log.debug(f"explanation: {explanation}")
            importances = explanation['data']['raw']['importances']

            log.debug(f"targetClass: {targetClassNames}")
            class_idx = 0

            list_kernel_shap = []
            for item in importances.items():
                log.debug(f"item: {item}")
                if item[0] == 'aggregated':
                    break
                data = dict(zip(item[1]['names'], item[1]['ranked_effect']))
                sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))


                list_importance = []
                original_list=list(sorted_data.values())
                min_val = min(original_list)
                max_val = max(original_list)
                normalized_list = [1 + 99 * (x - min_val) / (max_val - min_val) for x in original_list]
                sum_normalized = sum(normalized_list)
                scaling_factor = 100 / sum_normalized
                new_normalized_list = [x * scaling_factor for x in normalized_list]
                i=0
                for k in sorted_data.keys():
                    list_importance.append({"featureName": k, "importanceScore": round(new_normalized_list[i],4)})
                    i=i+1
                
                list_kernel_shap.append({"importantFeatures": list_importance})
                class_idx = class_idx + 1
                break

            return list_kernel_shap

        except Exception as e:
            log.error(e, exc_info=True)
            raise Exception
        
    def tree_shap(params: dict):
        log.info("Running Global Explanation for Structured Tabular Data using Tree Shap")
        try:
            categoricalFeatureNames = params["categoricalFeatureNames"]
            model = params['model']
            dataset = params['dataset']
            taskType = params['taskType']
            preprocessor = params['preprocessor']
            targetNames = params['targetNames']
            targetClassNames = params['targetClassNames']

            if targetClassNames is None:
                targetClassNames = list(dataset[targetNames].unique())

            category_map = None
            if categoricalFeatureNames is not None:
                category_map = gen_category_map(data=dataset, categorical_columns=categoricalFeatureNames)
            
            X = dataset.drop(targetNames, axis=1, inplace=False)

            column_names = list(X.columns.values)
            if params['featureNames'] is None:
                feature_names = column_names
            else:
                feature_names = params['featureNames']
                X = X[feature_names]
                column_names = list(X.columns.values)
            if taskType=="REGRESSION":
                tree_shap_explainer = TreeShap(
                    model,
                    task='regression',
                    feature_names=feature_names,
                    categorical_names=category_map
                )
            else:
                tree_shap_explainer = TreeShap(
                    model,
                    feature_names=feature_names,
                    categorical_names=category_map
                )
            tree_shap_explainer.fit()

            label_encoder = preprocessing.LabelEncoder()
            for col in column_names:
                X[col] = label_encoder.fit_transform(X[col])

            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer = imputer.fit(X)
            X = imputer.transform(X)

            explanation = tree_shap_explainer.explain(X[0:100])
            
            log.debug(f"explanation: {explanation}")
            importances = explanation['data']['raw']['importances']

            log.debug(f"targetClass: {targetClassNames}")
            class_idx = 0

            list_tree_shap = []
            for item in importances.items():
                log.info(f"item: {item}")
                if taskType == "REGRESSION":
                    if item[0] == 'aggregated':
                        continue
                else:                    
                    if item[0] == 'aggregated':
                        break
                data = dict(zip(item[1]['names'], item[1]['ranked_effect']))
                sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))

                list_importance = []
                original_list=list(sorted_data.values())
                min_val = min(original_list)
                max_val = max(original_list)
                normalized_list = [1 + 99 * (x - min_val) / (max_val - min_val) for x in original_list]
                sum_normalized = sum(normalized_list)
                scaling_factor = 100 / sum_normalized
                new_normalized_list = [x * scaling_factor for x in normalized_list]
                i=0
                for k in sorted_data.keys():
                    list_importance.append({"featureName": k, "importanceScore": round(new_normalized_list[i],4)})
                    i=i+1
                

                if taskType=="REGRESSION":
                    list_tree_shap.append({"importantFeatures": list_importance})
                else:
                    list_tree_shap.append({"importantFeatures": list_importance})
                class_idx = class_idx + 1
            log.debug(f"list_importance: {list_importance}")
            return list_tree_shap

        except Exception as e:
            log.error(e, exc_info=True)
            raise Exception

    def global_explain(dataset,
                       taskType,
                       method,
                       model,
                       preprocessor,
                       featureNames,
                       categoricalFeatureNames,
                       targetNames,
                       targetClassNames):
        try:
            
            params = {
                "dataset":dataset,
                "taskType": taskType,
                "model":model,
                "preprocessor":preprocessor,
                "featureNames": featureNames,
                "categoricalFeatureNames":categoricalFeatureNames,
                "targetNames":targetNames,
                "targetClassNames":targetClassNames}

            log.debug(f"method: {method}")

            dict_global_explain_mapping = {
                "PD-VARIANCE": ResponsibleAIExplain.partial_dependence_variance,
                "KERNEL-SHAP": ResponsibleAIExplain.kernel_shap,
                "TREE-SHAP": ResponsibleAIExplain.tree_shap,
                "PERMUTATION-IMPORTANCE":ResponsibleAIExplain.permutation_importance
            }

            return dict_global_explain_mapping[method](params)

        except Exception as e:
            log.error(e,exc_info=True)
            return [{"importantFeatures":[{"featureName": "", "importanceScore": ""}]}]
        
    def anchor_tabular_local_explanation(params: dict):
        # Log the start of the explanation process
        log.info("Running Local Explanation for Structured Tabular Data using Anchor Tabular")
        
        try:
            global url, input_request, output_response
            
            # Extract necessary information from params
            model = params['model']
            dataset = params['dataset']
            modelType = params['modelType']
            targetClassLabel = params['targetClassLabel']
            targetClassNames = params['targetClassNames']
            inputData = params['lineDataset']
            inputIndex = params['inputIndex']
            input_request = params['api_input_request']
            output_response = params['api_output_response']
            modelName = type(model).__name__

            # Check if the model is a pipeline
            if modelName =='Pipeline':
                model, dataset, targetClassLabel, inputData = ResponsibleAIExplain.pipeline_processing(model, dataset, targetClassLabel, inputData)

            # Check if the input index is valid
            if inputIndex < 0 or inputIndex >= len(dataset):
                raise ValueError("Input index must be between 0 and the length of the dataset.")

            # Define the prediction function based on the model type
            if modelType != 'API' and modelType is not None:
                predict_fn = lambda x: model.predict(x)
            elif modelType == 'API' and modelType is not None:
                url = model
                predict_fn = ResponsibleAIExplain.endpoint_check

            # Prepare the data and target
            data, target, featureNames, targetClassNames = ResponsibleAIExplain.prepare_data(dataset, targetClassLabel, targetClassNames)

            # Initialize the explainer
            explainer = AnchorTabular(predictor=predict_fn, feature_names=featureNames)

            # Fit the explainer
            explainer.fit(data)
            
            inp_data = inputData.values if inputData is not None else data[:500]

            result = []
            
            # Define a function to generate the explanation for each row in the DataFrame
            def explanation(inp_data):

                # Generate the explanation
                explaination = explainer.explain(np.array(inp_data), verbose_every=0)
                
                # Create a list of dictionaries directly from the data and feature names
                inputRow = [{"featureName": key, "featureValue": val} for key, val in zip(featureNames, inp_data.tolist())]

                # Make a prediction using the explainer's predictor
                prediction = targetClassNames[int(explainer.predictor(inp_data.to_numpy().reshape(1, -1))[0])]

                result.append({"inputRow": inputRow,
                               "modelPrediction": str(prediction),
                               "explanation": explaination.anchor})

            # Apply the function to each row in the DataFrame
            pd.DataFrame(inp_data).apply(explanation, axis=1)
            
            # Return the predicted target, anchor, and input row
            return [{"anchor": result,
                     "description": ANCHOR_TABULAR_DES}]

        except Exception as e:
            # Log the error and return an empty result
            log.error(e, exc_info=True)
            raise
        
    def kernel_shap_local_explanation(params: dict):
        # Log the start of the explanation process
        log.info("Running Local Explanation for Structured Tabular Data using Kernel Shap")
        
        try:
            global url, input_request, output_response

            # Extract the parameters
            model = params['model']
            taskType = params['taskType']
            modelType = params['modelType']
            dataset = params['dataset']
            targetClassLabel = params['targetClassLabel']
            targetClassNames = params['targetClassNames']
            inputIndex = params['inputIndex']
            inputData = params['lineDataset']
            input_request = params['api_input_request']
            output_response = params['api_output_response']
            modelName = type(model).__name__

            # Check if the model is a pipeline
            if modelName =='Pipeline':
                model, dataset, targetClassLabel, inputData = ResponsibleAIExplain.pipeline_processing(model, dataset, targetClassLabel, inputData)

            # Check if the input index is valid
            if inputIndex < 0 or inputIndex >= len(dataset):
                raise ValueError("Input index must be between 0 and the length of the dataset.")
            
            # Define the prediction function based on the model type
            if modelType != 'API':
                predict_fn = lambda x: model.predict(x)
            else:
                url = model
                predict_fn = ResponsibleAIExplain.endpoint_check

            # Prepare the data and target
            data, target, featureNames, targetClassNames = ResponsibleAIExplain.prepare_data(dataset, targetClassLabel, targetClassNames)

            # Initialize the explainer
            if taskType == 'REGRESSION':
                explainer = KernelShap(predict_fn, task='regression', feature_names=featureNames)
                
            elif taskType == 'CLASSIFICATION':
                explainer = KernelShap(predict_fn, feature_names=featureNames)
            
            # Check the size of the dataset
            if len(data) < 2:
                # If the dataset has less than 2 samples, it's too small to fit the explainer
                raise ValueError("Dataset is too small to fit the explainer. Please provide dataset with more samples.")
            elif len(data) > 100:
                # If the dataset has more than 100 samples, we only take the first 100 samples for efficiency
                train_data = data[:100]
            else:
                train_data = data
            
            # Fit the explainer
            explainer.fit(train_data)
            
            inp_data = inputData.values if inputData is not None else data[:500]
            result = []
            
            def explanation(data):
                # Generate the explanation
                explanation = explainer.explain(np.array([data]))

                # Extract feature importances from the explanation
                importances = explanation['data']['raw']['importances']

                # Initialize lists for storing feature importances and kernel SHAP values
                list_importance, sorted_data = [], {}
                
                # Iterate over the items in the importances dictionary
                for feature, importance in importances.items():
                    # Skip irrelevant features for regression tasks
                    if taskType == "REGRESSION" and feature == 'aggregated':
                        continue
                    # Skip irrelevant features for non-regression tasks
                    elif taskType != "REGRESSION" and feature != str(explanation['raw']['prediction'][0]):
                        continue

                    # Create a dictionary mapping feature names to their ranked effect
                    data_dict = dict(zip(importance['names'], importance['ranked_effect']))

                    # Sort the data by ranked effect in descending order
                    sorted_data = dict(sorted(data_dict.items(), key=lambda x: x[1], reverse=True))

                # Create a list of dictionaries directly from the data and feature names
                inputRow = [{"featureName": key, "featureValue": val} for key, val in zip(featureNames, data.tolist())]

                # Normalize the feature importances to a scale from 1 to 100
                original_list = list(sorted_data.values())
                min_val, max_val = min(original_list), max(original_list)
                normalized_list = [1 + 99 * (x - min_val) / (max_val - min_val) for x in original_list]

                # Scale the normalized importances so that they sum to 100
                scaling_factor = 100 / sum(normalized_list)
                new_normalized_list = [x * scaling_factor for x in normalized_list]

                # Create a list of dictionaries of feature names and their normalized importances
                list_importance = [{"featureName": k, "importanceScore": round(v, 4)} for k, v in zip(sorted_data.keys(), new_normalized_list)][:10]
                
                # Prepare the output based on the task type
                if modelType != "Keras":
                    predictedTarget = str(explanation['raw']['raw_prediction'][0]) if taskType == 'REGRESSION' else str(targetClassNames[explanation['raw']['raw_prediction'][0]])
                else:
                    predictedTarget = str(targetClassNames[int(np.argmax(explanation['raw']['raw_prediction'][0]))])
                
                result.append({"inputRow": inputRow,
                               "modelPrediction": predictedTarget,
                               "explanation": list_importance})

            # Apply the function to each row in the DataFrame
            pd.DataFrame(inp_data).apply(explanation, axis=1)
            
            return [{"importantFeatures": result,
                     "description":LOCAL_KERNEL_SHAP_DES}]
        
        except Exception as e:
            # Log the error and return an empty result
            log.error(e, exc_info=True)
            raise
        
    def kernel_shap_global_explanation(params: dict):
        # Log the start of the explanation process
        log.info("Running Global Explanation for Structured Tabular Data using Kernel Shap")
        
        try:
            global url,input_request,output_response
            
            # Extract necessary information from params
            model = params['model']
            taskType = params['taskType']
            modelType = params['modelType']
            dataset = params['dataset']
            targetClassLabel = params['targetClassLabel']
            targetClassNames = params['targetClassNames']
            input_request = params['api_input_request']
            output_response = params['api_output_response']
            modelName = type(model).__name__

            # Check if the model is a pipeline
            if modelName =='Pipeline':
                model, dataset, targetClassLabel, inputData = ResponsibleAIExplain.pipeline_processing(model, dataset, targetClassLabel)

            # Define the prediction function based on the model type
            if modelType != 'API':
                predict_fn = lambda x: model.predict(x)
            else:
                url = model
                predict_fn = ResponsibleAIExplain.endpoint_check

            # Prepare the data and target
            data, target, featureNames, targetClassNames = ResponsibleAIExplain.prepare_data(dataset, targetClassLabel, targetClassNames)

            # Initialize the explainer
            if taskType == 'REGRESSION':
                explainer = KernelShap(predict_fn, task='regression', feature_names=featureNames)
                
            elif taskType == 'CLASSIFICATION':
                explainer = KernelShap(predict_fn, feature_names=featureNames)
            
            elif taskType == 'CLUSTERING':
                # Train a decision tree on the data
                tree = DecisionTreeClassifier(max_depth=3)
                predictions = predict_fn(data)

                new_model = tree.fit(data, predictions)
                new_predict_fn = lambda x: new_model.predict(x)
                
                explainer = KernelShap(new_predict_fn, feature_names=featureNames)

            
            # Check the size of the dataset
            if len(data) < 2:
                # If the dataset has less than 2 samples, it's too small to fit the explainer
                raise ValueError("Dataset is too small to fit the explainer. Please provide dataset with more samples.")
            elif len(data) > 100:
                # If the dataset has more than 100 samples, we only take the first 100 samples for efficiency
                data = data[:100]
            
            # Fit the explainer
            explainer.fit(data)
            
            # Generate the explanation
            explanation = explainer.explain(data)

            # Extract feature importances from the explanation
            importances = explanation['data']['raw']['importances']

            # Initialize list for storing kernel SHAP values
            list_kernel_shap = []

            # Iterate over the items in the importances dictionary
            for item in importances.items():
                # Skip irrelevant features
                if item[0] == 'aggregated':
                    break

                # Create a dictionary mapping feature names to their ranked effect
                sorted_data = dict(sorted(zip(item[1]['names'], item[1]['ranked_effect']), key=lambda x: x[1], reverse=True))

                # Normalize the feature importances to a scale from 1 to 100
                original_list = list(sorted_data.values())
                min_val, max_val = min(original_list), max(original_list)
                normalized_list = [1 + 99 * (x - min_val) / (max_val - min_val) for x in original_list]

                # Scale the normalized importances so that they sum to 100
                scaling_factor = 100 / sum(normalized_list)
                new_normalized_list = [x * scaling_factor for x in normalized_list]

                # Define the threshold
                threshold = 5

                # Create a list of dictionaries of feature names and their normalized importances
                list_importance = [{"featureName": k, "importanceScore": round(v, 4)} for k, v in zip(sorted_data.keys(), new_normalized_list)]

                # Separate features based on the threshold
                important_features = [feature for feature in list_importance if feature['importanceScore'] >= threshold]
                other_features = [feature for feature in list_importance if feature['importanceScore'] < threshold]

                # Calculate the total importance score for 'Others'
                others_importance_score = sum(feature['importanceScore'] for feature in other_features)

                important_features.append({"featureName": "Others", "importanceScore": round(others_importance_score, 4)})

                # Append the list of important features to the list_kernel_shap list
                list_kernel_shap.append({"inputRow": None,
                                         "modelPrediction": None, 
                                         "explanation": important_features})
                break 
            
            return [{'importantFeatures': list_kernel_shap, 'featureNames': featureNames, 'description': GLOBAL_KERNEL_SHAP_DES}]
            
        except Exception as e:
            # Log the error and return an empty result
            log.error(e, exc_info=True)
            raise
        
    def tree_shap_local_explanation(params: dict):
        # Log the start of the explanation process
        log.info("Running Local Explanation for Structured Tabular Data using Tree Shap")
        
        try:
            global url
            
            # Extract necessary information from params
            model = params['model']
            taskType = params['taskType']
            dataset = params['dataset']
            targetClassLabel = params['targetClassLabel']
            targetClassNames = params['targetClassNames']
            inputData = params['lineDataset']
            inputIndex = params['inputIndex']
            modelName = type(model).__name__

            # Check if the model is a pipeline
            if modelName =='Pipeline':
                model, dataset, targetClassLabel, inputData = ResponsibleAIExplain.pipeline_processing(model, dataset, targetClassLabel, inputData)

            #Check if the input index is valid
            if inputIndex < 0 or inputIndex >= len(dataset):
                raise ValueError("Input index must be between 0 and the length of the dataset.")

            # Prepare the data and target
            data, target, featureNames, targetClassNames = ResponsibleAIExplain.prepare_data(dataset, targetClassLabel, targetClassNames)

            # Initialize the explainer
            if taskType == 'REGRESSION':
                explainer = TreeShap(model, task='regression', feature_names=featureNames)
                
            elif taskType == 'CLASSIFICATION':
                explainer = TreeShap(model, feature_names=featureNames)

            # Fit the explainer
            explainer.fit()
            
            inp_data = inputData.values if inputData is not None else data[:500]
            result = []

            def explanation(data):

                # Generate the explanation 
                explanation = explainer.explain(np.array([data]), check_additivity=False)

                # Extract feature importances from the explanation
                importances = explanation['data']['raw']['importances']

                # Iterate over the items in the importances dictionary
                for item in importances.items():
                    
                    if modelName in ['XGBClassifier','GradientBoostingClassifier']:
                        # Prepare the predicted target based on the task type
                        predictedTarget = str(explanation['raw']['raw_prediction']) if taskType == 'REGRESSION' else str(targetClassNames[explanation['raw']['prediction']])
                        # Skip irrelevant features
                        if (taskType == "REGRESSION" and item[0] == 'aggregated') or (taskType != "REGRESSION" and item[0] != str(explanation['raw']['prediction']) and item[0] != 'aggregated'):
                            continue
                    else:
                        # Prepare the predicted target based on the task type
                        predictedTarget = str(explanation['raw']['raw_prediction']) if taskType == 'REGRESSION' else str(targetClassNames[explanation['raw']['prediction'][0]])
                        if (taskType == "REGRESSION" and item[0] == 'aggregated') or (taskType != "REGRESSION" and item[0] != str(explanation['raw']['prediction'][0])):
                            continue
                    
                    # Create a sorted dictionary of feature names and their ranked effects
                    sorted_data = dict(sorted(zip(item[1]['names'], item[1]['ranked_effect']), key=lambda x: x[1], reverse=True))
                
                # Create a list of dictionaries directly from the data and feature names
                inputRow = [{"featureName": key, "featureValue": val} for key, val in zip(featureNames, data.to_numpy().tolist())]

                # Normalize the feature importances to a scale from 1 to 100
                original_list = list(sorted_data.values())
                min_val, max_val = min(original_list), max(original_list)
                normalized_list = [1 + 99 * (x - min_val) / (max_val - min_val) for x in original_list]
                
                # Scale the normalized importances so that they sum to 100
                scaling_factor = 100 / sum(normalized_list)
                new_normalized_list = [x * scaling_factor for x in normalized_list]

                # Create a list of dictionaries of feature names and their normalized importances
                list_importance = [{"featureName": k, "importanceScore": round(v, 4)} for k, v in zip(sorted_data.keys(), new_normalized_list)]
                
                # Append the list of important features to the list_tree_shap list
                result.append({"inputRow": inputRow, 
                               "modelPrediction": predictedTarget, 
                               "explanation": list_importance})
            
            # Apply the function to each row in the DataFrame
            pd.DataFrame(inp_data).apply(explanation, axis=1)
            
            return [{"importantFeatures": result,
                    "description":LOCAL_TREE_SHAP_DES}]
        
        except Exception as e:
            # Log the error and return an empty result
            log.error(e, exc_info=True)
            raise
        
    def tree_shap_global_explanation(params: dict):
        log.info("Running Global Explanation for Structured Tabular Data using Tree Shap")
        try:
            # Extract necessary information from params
            model = params['model']
            taskType = params['taskType']
            dataset = params['dataset']
            targetClassLabel = params['targetClassLabel']
            targetClassNames = params['targetClassNames']
            modelName = type(model).__name__

            # Check if the model is a pipeline
            if modelName =='Pipeline':
                model, dataset, targetClassLabel, inputData = ResponsibleAIExplain.pipeline_processing(model, dataset, targetClassLabel)
            
            # Define the prediction function for the model
            predict_fn = lambda x: model.predict(x)

            # Prepare the data and target
            data, target, featureNames, targetClassNames = ResponsibleAIExplain.prepare_data(dataset, targetClassLabel, targetClassNames)

            # Initialize the explainer
            if taskType == 'REGRESSION':
                explainer = TreeShap(model, task='regression', feature_names=featureNames)
                
            elif taskType == 'CLASSIFICATION':
                explainer = TreeShap(model, feature_names=featureNames)
            
            elif taskType == 'CLUSTERING':
                tree = DecisionTreeClassifier(max_depth=3)
                predictions = predict_fn(data)
                
                new_model = tree.fit(data, predictions)
                
                explainer = TreeShap(new_model, feature_names=featureNames, categorical_names=targetClassNames)
            
            # Fit the explainer
            explainer.fit()

            # Check if the dataset is too large 
            if len(data) > 100:
                # If the dataset is too large, limit it to the first 100 samples
                data = data[:100]
            elif len(data) < 2:
                # If the dataset is too small, raise an error
                raise ValueError("Dataset is too small to fit the explainer. Please provide dataset with more samples.")
            
            # Generate the explanation for the first 100 samples
            explanation = explainer.explain(data)
            
            # Extract feature importances from the explanation
            feature_importances = explanation['data']['raw']['importances']

            # Initialize list for storing tree SHAP values
            shap_values, feature_importance_list = [], []
            
            idx = 0
            # Iterate over the items in the feature_importances dictionary
            for idx, (class_name, feature_importance) in enumerate(feature_importances.items()):
                # Skip irrelevant features
                if class_name == 'aggregated' and taskType == "REGRESSION":
                    continue
                if taskType == "CLASSIFICATION" and class_name != 'aggregated':
                    continue
                
                # Create a sorted dictionary of feature names and their ranked effects
                sorted_importances = dict(sorted(zip(feature_importance['names'], feature_importance['ranked_effect']), key=lambda x: x[1], reverse=True))
                
                # Normalize the feature importances to a scale from 1 to 100
                original_importances = list(sorted_importances.values())
                min_importance, max_importance = min(original_importances), max(original_importances)
                normalized_importances = [1 + 99 * (x - min_importance) / (max_importance - min_importance) for x in original_importances]

                # Scale the normalized importances so that they sum to 100
                scaling_factor = 100 / sum(normalized_importances)
                scaled_importances = [x * scaling_factor for x in normalized_importances]

                feature_importance_dict = []
                # Add the target class name to the feature importance dictionary
                if class_name == 'aggregated':
                    feature_importance_dict.append({"ClassName": "Overall aggregated feature importance"})
                else:
                    if isinstance(targetClassNames[idx], str):
                        feature_importance_dict.append({"ClassName": targetClassNames[idx]})
                    else:
                        feature_importance_dict.append({"ClassName": targetClassNames[idx].tolist()})

                # Create a list of dictionaries of feature names and their normalized importances
                feature_importance_values_dict = [{"featureName": name, "importanceScore": round(value, 4)} for name, value in zip(sorted_importances.keys(), scaled_importances)]

                feature_importance_dict[0]["importantFeatures"] = feature_importance_values_dict

                # Add the feature importance dictionary to the list
                feature_importance_list.append(feature_importance_dict[0])

            # Append the list of important features to the shap_values list
            shap_values.append({"inputRow": None,
                                "modelPrediction": None, 
                                "explanation": feature_importance_list})

            # Return the list of important features
            return [{'importantFeatures': shap_values, 'description': GLOBAL_TREE_SHAP_DES}]

        except Exception as e:
            log.error(f"Failed to generate global explanation: {str(e)}")
            raise
        
    def permutation_importance_global_explanation(params: dict):
        # Log the start of the explanation process
        log.info("Running Global Explanation for Structured Tabular Data using Permutation Importance")
        
        try:
            global url,input_request,output_response
            
            # Extract necessary information from params
            model = params['model']
            taskType = params['taskType']
            modelType = params['modelType']
            dataset = params['dataset']
            targetClassLabel = params['targetClassLabel']
            targetClassNames = params['targetClassNames']
            input_request = params['api_input_request']
            output_response = params['api_output_response'] 
            modelName = type(model).__name__

            # Check if the model is a pipeline
            if modelName =='Pipeline':
                model, dataset, targetClassLabel, inputData = ResponsibleAIExplain.pipeline_processing(model, dataset, targetClassLabel)

            # Define the prediction function based on the model type
            if modelType != 'API':
                predict_fn = lambda x: model.predict(x)
            else:
                url = model
                predict_fn = ResponsibleAIExplain.endpoint_check

            # Prepare the data and target
            data, target, featureNames, targetClassNames = ResponsibleAIExplain.prepare_data(dataset, targetClassLabel, targetClassNames)

            # Initialize the explainer
            if taskType == 'REGRESSION':
                explainer = PermutationImportance(predictor=predict_fn, loss_fns=['mean_absolute_error'], feature_names=featureNames)
                
            elif taskType == 'CLASSIFICATION':
                explainer = PermutationImportance(predictor=predict_fn, score_fns=['accuracy'], feature_names=featureNames)

            # Generate the explanation
            exp_importance = explainer.explain(X=data, y=target)

            # Extract feature importance values and create a dictionary with feature names
            feature_importance_values = [item['mean'] for item in exp_importance.feature_importance[0]]
            data = dict(zip(featureNames, feature_importance_values))

            # Sort the data by importance values in descending order
            sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))

            # Normalize the importance values to a scale from 1 to 100
            original_list = list(sorted_data.values())
            min_val, max_val = min(original_list), max(original_list)
            normalized_list = [1 + 99 * (x - min_val) / (max_val - min_val) for x in original_list]

            # Scale the normalized importances so that they sum to 100
            scaling_factor = 100 / sum(normalized_list)
            new_normalized_list = [x * scaling_factor for x in normalized_list]

            # Create a list of dictionaries of feature names and their normalized importances
            list_data = [{"featureName": k, "importanceScore": round(v, 4)} for k, v in zip(sorted_data.keys(), new_normalized_list)]
            
            # Return the list of important features
            return [{'importantFeatures': [{"inputRow": None,
                                            "modelPrediction": None, 
                                            "explanation": list_data}],
                     "description": GLOBAL_PERMUTATION_IMPORTANCE_DES}]

        except Exception as e:
            # Log the error and return an empty result
            log.error(e, exc_info=True)
            raise
        
    def partial_dependence_variance_global_explanation(params: dict):
        # Log the start of the explanation process
        log.info("Running Global Explanation for Structured Tabular Data using Partial Dependence Variance")
        
        try:
            global url,input_request,output_response
            
            # Extract necessary information from params
            model = params['model']
            taskType = params['taskType']
            modelType = params['modelType']
            dataset = params['dataset']
            targetClassLabel = params['targetClassLabel']
            targetClassNames = params['targetClassNames']
            input_request = params['api_input_request']
            output_response = params['api_output_response']
            modelName = type(model).__name__

            # Check if the model is a pipeline
            if modelName =='Pipeline':
                model, dataset, targetClassLabel, inputData = ResponsibleAIExplain.pipeline_processing(model, dataset, targetClassLabel)

            # Define the prediction function based on the model type
            if modelType != 'API':
                # predict_fn = model.predict
                predict_fn = lambda x: model.predict(x)
            else:
                url = model
                predict_fn = ResponsibleAIExplain.endpoint_check

            # Prepare the data and target
            data, target, featureNames, targetClassNames = ResponsibleAIExplain.prepare_data(dataset, targetClassLabel, targetClassNames)

            # Initialize the explainer
            explainer = PartialDependenceVariance(predictor=predict_fn, 
                                                  target_names=targetClassNames, 
                                                  feature_names=featureNames
                                                  )
           
            # Generate the explanation
            exp_importance = explainer.explain(X=data, method='importance')
        
            # Create a dictionary with feature names and their importance values
            data = dict(zip(featureNames, exp_importance.feature_importance[0]))
            
            # Sort the data by importance values in descending order
            sorted_data = dict(sorted(data.items(), key = lambda x: x[1], reverse = True))

            # Normalize the importance values to a scale from 1 to 100
            original_list = list(sorted_data.values())
            min_val, max_val = min(original_list), max(original_list)
            normalized_list = [1 + 99 * (x - min_val) / (max_val - min_val) for x in original_list]

            # Scale the normalized importances so that they sum to 100
            scaling_factor = 100 / sum(normalized_list)
            new_normalized_list = [x * scaling_factor for x in normalized_list]

            # Create a list of dictionaries of feature names and their normalized importances
            list_data = [{"featureName": k, "importanceScore": round(v, 4)} for k, v in zip(sorted_data.keys(), new_normalized_list)]

            # Return the list of important features
            return [{'importantFeatures': [{"inputRow": None,
                                            "modelPrediction": None, 
                                            "explanation": list_data}],
                     "description": GLOBAL_PD_VARIANCE_DES}]
            
        except Exception as e:
            # Log the error and return an empty result
            log.error(e, exc_info=True)
            raise
        
    def timeSeries_local_explanation(params: dict):
        # Log the start of the explanation process
        log.info("Running Local Explanation for Structured Tabular Data using Time Series")
        
        try:
            # Extract necessary information from params
            model = params['model']
            dataset = params['dataset']
            targetClassLabel = params['targetClassLabel']
            targetClassNames = params['targetClassNames']
            inputIndex = params['inputIndex']

            # Prepare the data and target
            data, target, featureNames, targetClassNames = ResponsibleAIExplain.prepare_data(dataset, targetClassLabel, targetClassNames)

            # Initialize the forecasting model
            forecasting_model = Forecaster(model=model, forecast_function="predict")
            
            # Define the relevant history
            relevant_history = 25
            
            # Initialize the explainer
            explainer = TSLimeExplainer(model = functools.partial(forecasting_model.predict),
                                        input_length = 120,
                                        relevant_history = relevant_history,
                                        perturbers = [BlockBootstrapPerturber(window_length=min(10, 120-1), block_length=2, block_swap=2)],
                                        n_perturbations = 10000,
                                        random_seed = 22
                                        )

            instance = data[inputIndex:inputIndex+120]

            ts_instance = tsFrame(instance)
            ts_instance.index = pd.to_numeric(ts_instance.index)
            explanation = explainer.explain_instance(ts_instance)

            selected_feature_columns = featureNames
            selected_feature_names = featureNames
            lent = len(selected_feature_columns)
            lent = lent / 3
            fig = plt.figure(layout='constrained', figsize=(12,8))
            gs0 = gridspec.GridSpec(2, 1, figure=fig, height_ratios= [1, 3], top=0.9)
            gs2 = gridspec.GridSpecFromSubplotSpec(int(lent)+1, 3, subplot_spec=gs0[1])
            gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[0], width_ratios=[4.5, 10, 4.5])

            def plot_feature(figure, scores, feature_values, gs, title, legend=False):
                ax = fig.add_subplot(gs)
                ax.plot(feature_values, label='Input Time Series', marker='o')
                ax.set_title(title, fontsize=10)

                ax.bar(feature_values.index, scores, 0.4, label = 'TSLime Weights (Normalized)', color='red')
                ax.axhline(y=0, color='r', linestyle='-', alpha=0.4)
                if legend:
                    ax.legend(bbox_to_anchor=(1.5, 1.0), loc='upper right', fontsize=8)

            instance_prediction = explanation['model_prediction'][0]
            normalized_weights = (explanation['history_weights'] / np.mean(np.abs(explanation['history_weights'])))

            for i, feature_col in enumerate(selected_feature_columns):
                if i > 0:
                    gs = gs2[i-1]
                else:
                    gs = gs1[1]
                normalized_weights = (explanation['history_weights'] / np.mean(np.abs(explanation['history_weights'])))
                sc= normalized_weights[:,i].flatten()
            
                feature_v=ts_instance.iloc[-relevant_history:, i]
                feature_range = max(np.max(feature_v), - np.min(feature_v))
                weights_range = max(np.max(sc), - np.min(sc))
            
                scaling_factor = feature_range / weights_range
                normalized_weights = normalized_weights * scaling_factor
                plot_feature(figure=fig,
                            scores=normalized_weights[:,i].flatten(), 
                            feature_values=ts_instance.iloc[-relevant_history:, i],
                            gs=gs,
                            title="{}".format(feature_col),
                            legend=(i==0))

            fig.suptitle("Time Series Lime Explanation Plot with forecast={}".format(str(instance_prediction)), fontsize="x-large")
            l1 = []
            l1.append(fig)
            list_time_series=[]

            list_time_series.append({"modelPrediction": str(instance_prediction),
                                    "timeSeries": l1})

            return [{"timeSeries":list_time_series,
                     "featureNames":featureNames,
                     "description":LOCAL_TS_LIME_EXPLAINER_DES}]
            
        except Exception as e:
            # Log the error and re-raise the exception
            log.error(e, exc_info=True)
            raise
        
    def timeSeries_local_shap_explanation(params: dict):
        # Log the start of the explanation process
        log.info("Running Local Explanation for Structured Tabular Data using kernel shap")
        
        try:
            # Extract necessary information from params
            model = params['model']
            dataset = params['dataset']
            targetClassLabel = params['targetClassLabel']

            date_column_name = ResponsibleAIExplain.find_date_column(dataset)
            dataset.set_index(date_column_name, inplace=True)

            featureNames = dataset.columns.to_list()
            featureNames.remove(targetClassLabel)
            exog = dataset[featureNames]
            
            def predict_arimax(exog_data):
                # Ensure the exogenous data has the same structure as the training data
                exog_data = pd.DataFrame(exog_data, columns=featureNames)
                predictions = model.get_forecast(steps=len(exog_data), exog=exog_data).predicted_mean
                
                return predictions.values
            
            explainer = shap.KernelExplainer(predict_arimax, exog)
            shap_values = explainer.shap_values(exog)
            
            figs = []
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, exog, plot_type='bar', show=False)
            figs.append(fig)

            prediction = predict_arimax(exog)[0]
            arima_time_series = []
            
            arima_time_series.append({"modelPrediction": str(prediction),
                                    "timeSeries": figs})

            return [{"timeSeries":arima_time_series,
                     "description":LOCAL_KERNEL_SHAP_DES}]

        except Exception as e:
            # Log the error and re-raise the exception
            log.error(e, exc_info=True)
            raise
    
    def timeseries_lime_tabular_explaination(params: dict):
        # Log the start of the explanation process
        log.info("Running Local Explanation for Structured Tabular Data using lime tabular")
        
        try:
            # Extract necessary information from params
            model = params['model']
            dataset = params['dataset']
            targetClassLabel = params['targetClassLabel']
            
            date_column_name = ResponsibleAIExplain.find_date_column(dataset)
            dataset.set_index(date_column_name, inplace=True)
  
            featureNames = dataset.columns.to_list()
            featureNames.remove(targetClassLabel)
            endog = dataset[targetClassLabel]
            exog = dataset[featureNames]

            def predict_arimax(exog_data):
                # Ensure the exogenous data has the same structure as the training data
                exog_data = pd.DataFrame(exog_data, columns=featureNames)
                predictions = model.get_forecast(steps=len(exog_data), exog=exog_data).predicted_mean
                
                return predictions.values

            explainer = lime.lime_tabular.LimeTabularExplainer(exog.values, mode="regression", training_labels=endog, feature_names=featureNames)

             # Explain predictions for each sample in the subset
            explanation = explainer.explain_instance(exog.values[0], predict_arimax, num_features= len(featureNames))

            # Save the explanation to a temporary file
            figs = []
            fig = explanation.as_pyplot_figure()
            plt.tight_layout()
            plt.close(fig)

            figs.append(fig)
            
            arima_time_series = []

            arima_time_series.append({"inputRow": [{"featureName": key, "featureValue": val} for key, val in zip(featureNames, exog.values[0])],
                                    "modelPrediction": str(explanation.predicted_value),
                                    "timeSeries": figs})

            return [{"timeSeries":arima_time_series,
                     "description":LIME_TABULAR_DES}]
            
        except Exception as e:
            # Log the error and re-raise the exception
            log.error(e, exc_info=True)
            raise    
    
    def lime_tabular_local_explanation(params: dict):
        # Log the start of the explanation process
        log.info("Running Local Explanation for Structured Tabular Data using Lime Tabular")
        
        try:
            global url
            
            # Extract necessary information from params
            model = params['model']
            taskType = params['taskType']
            modelType = params['modelType']
            dataset = params['dataset']
            targetClassLabel = params['targetClassLabel']
            targetClassNames = params['targetClassNames']
            inputData = params['lineDataset']
            inputIndex = params['inputIndex']
            modelName = type(model).__name__

             # Check if the model is a pipeline
            if modelName =='Pipeline':
                model, dataset, targetClassLabel, inputData = ResponsibleAIExplain.pipeline_processing(model, dataset, targetClassLabel, inputData)

            # Check if the input index is valid
            if inputIndex < 0 or inputIndex >= len(dataset):
                raise ValueError("Input index must be between 0 and the length of the dataset.")

            # Prepare the data and target
            data, target, featureNames, targetClassNames = ResponsibleAIExplain.prepare_data(dataset, targetClassLabel, targetClassNames)

            # Initialize the explainer
            if taskType == 'REGRESSION':
                explainer = LimeTabularExplainer(data, feature_names=featureNames, class_names=targetClassNames, mode='regression')
                
            elif taskType == 'CLASSIFICATION':
                explainer = LimeTabularExplainer(data, feature_names=featureNames, class_names=targetClassNames, mode='classification')

            inp_data = inputData.values if inputData is not None else data[:500]
            result = []
            
            def explanation(data):
                # Generate the explanation
                if taskType == 'REGRESSION':
                    explanation = explainer.explain_instance(data, model.predict)
                    prediction = round(explanation.predicted_value, 5)
                else:
                    if modelType != 'Keras':
                        explanation = explainer.explain_instance(data, model.predict_proba)
                        prediction = targetClassNames[model.predict([data])[0]]
                    else:
                        explanation = explainer.explain_instance(data, model.predict)
                        prediction = targetClassNames[int(np.argmax(model.predict([data])[0]))]
                
                data_dict = dict(explanation.as_list())
                # Sort the data by importance values in descending order
                sorted_data = dict(sorted(data_dict.items(), key=lambda x: abs(x[1]), reverse=True))
                # Normalize the absolute importance values to a scale from 1 to 100
                original_list = list(sorted_data.values())

                abs_list = [abs(x) for x in original_list]
               
                min_val, max_val = min(abs_list), max(abs_list)
               
                if max_val == min_val:
                    normalized_list = [1 for _ in abs_list]
                else:
                    normalized_list = [1 + 99 * (x - min_val) / (max_val - min_val) for x in abs_list]

                # Reapply the original signs to the normalized values
                signed_normalized_list = [x if original >= 0 else -x for x, original in zip(normalized_list, original_list)]

                # Scale the normalized importances so that they sum to 100
                scaling_factor = 100 / sum(abs(x) for x in signed_normalized_list)
                new_normalized_list = [x * scaling_factor for x in signed_normalized_list]
                
                # Generate the list of importances using a list comprehension with scaled scores
                list_importance = [{"featureName": k, "importanceScore": round(v, 4)} for k, v in zip(sorted_data.keys(), new_normalized_list)][:10]
               
                # Create a list of dictionaries directly from the data and feature names
                inputRow = [{"featureName": key, "featureValue": val} for key, val in zip(featureNames, data.to_numpy().tolist())][:10]
                
                result.append({"modelPrediction": str(prediction),
                                "explanation":list_importance,
                                "inputRow":inputRow})
                
            # Apply the function to each row in the DataFrame
            pd.DataFrame(inp_data).apply(explanation, axis=1)
            
            return [{"importantFeatures":result,
                     "featureNames": featureNames,
                     "description":LIME_TABULAR_DES}]

        except Exception as e:
            # Log the error and return an empty result
            log.error(e, exc_info=True)
            raise
        
    def anchor_text_local_explanation(params: dict):
        try:
            # Extract necessary information from params
            model = params['model']
            dataset = params['dataset']
            preprocessor = params['preprocessor']
            targetClassNames = params['targetClassNames']
            inputData = params['lineDataset']
        
            if preprocessor is not None:
                predict_fn = lambda x: model.predict(preprocessor.transform(x))
            else:
                predict_fn = lambda x: model.predict(x)

            explainer = AnchorText.load('../models//alibi//', predictor=predict_fn)
            result=[]
            
            def process_row(row):
                inputText = row  # assuming row is a text
                prediction = targetClassNames[predict_fn([inputText])[0]]
                
                explanation = explainer.explain(inputText,
                                                threshold=0.95,
                                                stop_on_first=True,
                                                min_samples_start=100,
                                                coverage_samples=100
                                                )
               
                if len(explanation.anchor) == 0 :
                    url = os.getenv("TOKEN_IMP_URL")
                    payload = {"inputPrompt": inputText, "modelName": "code" }
                    response = requests.request("POST", url, json=payload, verify=False).json()
                    exp = response['token_importance_mapping'][0]['token'].replace(' ', '')
                    if exp in inputText:
                        explanation.anchor = [word for word in inputText.split() if exp in word]

                # Check if explanation.anchor length is more than one and does not contain certain keywords
                modelConfidence = 'Low'
                if len(explanation.anchor) > 1 and not any(keyword in list(map(str.lower, explanation.anchor)) for keyword in ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'shall', 'would', 'should', 'can', 'could', 'may', 'might', 'must', 'ought','need', 'used', 'to', 'of', 'in', 'on','with']):
                    modelConfidence = 'High'
                elif len(explanation.anchor) == 1 and not any(keyword in list(map(str.lower, explanation.anchor)) for keyword in ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'shall', 'would', 'should', 'can', 'could', 'may', 'might', 'must', 'ought','need', 'used', 'to', 'of', 'in', 'on','with']):
                    modelConfidence = 'Moderate'                    
                elif len(explanation.anchor) == 1 and any(keyword in list(map(str.lower, explanation.anchor)) for keyword in ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'shall', 'would', 'should', 'can', 'could', 'may', 'might', 'must', 'ought','need', 'used', 'to', 'of', 'in', 'on','with']) or (len(explanation.anchor) == 0):
                    modelConfidence = 'Low'
                result.append({"inputText": inputText, "modelPrediction":prediction, "modelConfidence": modelConfidence, "explanation":explanation.anchor})
            
            dataset = inputData if inputData is not None else dataset[:500]
            # Apply the function to each row in the DataFrame
            dataset[dataset.columns[0]].apply(process_row)
            
            return [{"anchor": result, "description": ANCHOR_TABULAR_DES}]

        except Exception as e:
            # Log the error and re-raise the exception
            log.error(e, exc_info=True)
            raise

    def integrated_gradients_text_local_explanation(params: dict):
        log.info("Running Local Explanation for Unstructured Text Data using Integrated Gradients")
        try:

            model = params['model']
            dataset = params['dataset']
            preprocessor = params['preprocessor']
            targetClassNames = params['targetClassNames']
            
            result=[]
            def process_row(row):
                
                inputText = row
                if preprocessor is not None:
                    tokenized_sentence = preprocessor.texts_to_sequences([inputText])
                    preprocessed_sentence = pad_sequences(tokenized_sentence, padding='post', maxlen=100)
                else:
                    preprocessed_sentence = inputText
                predictions = ((model(preprocessed_sentence).numpy() > 0.5).astype('int32'))[0] # model(preprocessed_sentence).numpy().argmax(axis=1)
               
                layer = model.get_layer('embedding')
                explainer  = IntegratedGradients(model, 
                                                layer=layer, 
                                                n_steps=50, 
                                                method="gausslegendre", 
                                                internal_batch_size=100)
                explanation = explainer.explain(preprocessed_sentence, 
                                                baselines=None, 
                                                target=predictions, 
                                                attribute_to_layer_inputs=False)
                
                # Get attributions values from the explanation object
                attrs = explanation.attributions[0]
                attrs = attrs.sum(axis=2)
                prediction = targetClassNames[predictions[0]]

                words = inputText.split(" ")
                colors = ResponsibleAIExplain.colorize(attrs[0])
                result.append({"inputText":inputText,"modelConfidence": 'Moderate', "prediction": prediction, "explanation": "".join(list(map(ResponsibleAIExplain.hlstr, words, colors)))})
            # if dataset is not None:
            dataset[dataset.columns[0]].apply(process_row) 

            return [{"attributionsText": result, "description": INTEGRATED_GRADIENT_DES}]

        except Exception as e:
            log.error(e, exc_info=True)
            raise
        
    def get_explanation(model=None, taskType=None, modelType=None, dataset=None, 
                    preprocessor=None, targetClassLabel=None, 
                    targetClassNames=None, method=None, scope=None, lineDataset=None, inputIndex=None,
                    api_input_request=None, api_output_response=None):
        try:
            # Prepare the parameters
            params = {
                        "model": model,
                        "taskType": taskType,
                        "modelType": modelType,  
                        "dataset": dataset,
                        "preprocessor": preprocessor,
                        "targetClassLabel": targetClassLabel,
                        "targetClassNames": targetClassNames,
                        "method": method,
                        "scope": scope,
                        "lineDataset": lineDataset,
                        "inputIndex": inputIndex,
                        "api_input_request": api_input_request,
                        "api_output_response": api_output_response
                        #"prompt": prompt
                        }
            
            # Define the mapping between methods and their corresponding functions
            dict_local_explain_mapping = {
                "LOCAL" : {
                            "ANCHOR-TABULAR": ResponsibleAIExplain.anchor_tabular_local_explanation,
                            "KERNEL-SHAP": ResponsibleAIExplain.kernel_shap_local_explanation,
                            "TREE-SHAP": ResponsibleAIExplain.tree_shap_local_explanation,
                            "LIME-TABULAR": ResponsibleAIExplain.lime_tabular_local_explanation,
                            "TS-LIMEEXPLAINER": ResponsibleAIExplain.timeSeries_local_explanation,
                            "TS-SHAPEXPLAINER": ResponsibleAIExplain.timeSeries_local_shap_explanation,
                            "TS-LIME-TABULAR-EXPLAINER": ResponsibleAIExplain.timeseries_lime_tabular_explaination,
                            "ANCHOR-TEXT": ResponsibleAIExplain.anchor_text_local_explanation,
                            "INTEGRATED-GRADIENTS": ResponsibleAIExplain.integrated_gradients_text_local_explanation
                        },
                "GLOBAL": {
                            "KERNEL-SHAP": ResponsibleAIExplain.kernel_shap_global_explanation,
                            "TREE-SHAP": ResponsibleAIExplain.tree_shap_global_explanation,
                            "PERMUTATION-IMPORTANCE": ResponsibleAIExplain.permutation_importance_global_explanation,
                            "PD-VARIANCE": ResponsibleAIExplain.partial_dependence_variance_global_explanation
                        }
            }
            
            # Call the corresponding function based on the method and return the result
            return dict_local_explain_mapping[scope][method](params)

        except Exception as e:
            # Log the error and re-raise the exception
            log.error(e, exc_info=True)
            raise
