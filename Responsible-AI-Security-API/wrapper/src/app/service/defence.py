'''
MIT license https://opensource.org/licenses/MIT
Copyright 2024 Infosys Ltd
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''




import os
import pandas as pd
import numpy as np
import csv
import json
import pickle

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from app.service.utility import Utility as UT
from app.config.logger import CustomLogger
import concurrent.futures as con


telemetry_flg =os.getenv("TELEMETRY_FLAG")

apiEndPoint ='/v1/security/model'
errorRequestMethod = 'GET'

log = CustomLogger()

class Defence:


    def generateDenfenseModel(payload):
        
        try:
            payload_folder_path = UT.getcurrentDirectory() + "/database/payload"
            payload_path = ""
            SAFE_DIR = payload_folder_path
            def open_safe_file(filename):
                if '..' in filename or '/' in filename:
                    raise ValueError("Invalid filename")
                payload_path = os.path.join(SAFE_DIR, filename+".txt")
                return open(os.path.join(SAFE_DIR, filename+".txt"), "w", newline="")
            
            with open_safe_file(payload["modelName"]) as f:
                data = f.read()
            data = json.loads(data)
            Output_column = data["groundTruthClassLabel"]

            #reading original dataset
            data_path = UT.getcurrentDirectory()+f'/database/data/{payload["dataFileName"]}.csv'
            df1 = pd.read_csv(data_path, delimiter=',')
            df1.drop(Output_column,axis=1,inplace=True)
            df1.insert(df1.shape[1],"Attack",[0 for i in range(df1.shape[0])])

            #reading attack dataset
            # root_path = os.getcwd()[:-4] + "/database/report"
            # report_folder = os.path.join(root_path,payload['folderName'])
            # report_csv = os.path.join(report_folder,'Attack_Samples.csv')
            # df2 = pd.read_csv(report_csv)
            # cols = list(df2.columns)
            # if len(cols) > 4:
            #     for col in ['target', 'prediction', 'result', 'label', 'Index']:
            #         if col in cols:
            #             df2.drop(col, axis=1, inplace=True)
            # else:
            #     pass
            # df2.insert(df2.shape[1],"Attack",[1 for i in range(df2.shape[0])])

            #reading attack dataset
            root_path = UT.getcurrentDirectory() + "/database/report"
            report_folder = os.path.join(root_path,payload['folderName'])
            report_csv = os.path.join(report_folder,'Attack_Samples.csv')
            df2 = pd.read_csv(report_csv)
            df2.drop(Output_column,axis=1,inplace=True)
            df2.drop(columns=df2.columns[-2:], axis=1, inplace=True)
            cols = list(df2.columns)
            df2.insert(df2.shape[1],"Attack",[1 for i in range(df2.shape[0])])
    
            #creating defensemodel dataset
            fields=[]
            for col in df1.columns:
                fields.append(col)
            df1=np.array(df1)
            df2=np.array(df2)

            #creating defense_model path
            root_path = UT.getcurrentDirectory() + "/database"
            report_path = os.path.join(root_path+"/report",payload['folderName'])
            temp_path = root_path + "/cacheMemory"

            SAFE_DIR = temp_path
            def open_safe_file(filename):
                if '..' in filename or '/' in filename:
                    raise ValueError("Invalid filename")
                temp_path = os.path.join(SAFE_DIR, filename)
                return open(os.path.join(SAFE_DIR, filename),"w",newline="")
            with open_safe_file(f'{payload["modelName"]}defenseModel.csv') as f:
                write = csv.writer(f)
                write.writerow(fields)
                write.writerows(df1)
                write.writerows(df2)
        
            # creating defense Model
            df=pd.read_csv(temp_path)
            X=df.loc[:,df.columns!="Attack"]
            Y=df["Attack"]
            X_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1)
            model=XGBClassifier()
            model.fit(X_train,y_train)
            pickle.dump(model, open(os.path.join(report_path,"DefenseModel.pkl"), 'wb'))

            del model,X_train,x_test,y_train,y_test,df1,df2
            UT.databaseDelete(temp_path)
        except Exception as exc:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "generateDenfenseModel", exc, apiEndPoint, errorRequestMethod)
            raise Exception


    def generateCombinedDenfenseModel(payload):

        try:
            root_path = os.getcwd()
            root_path = UT.getcurrentDirectory() + "/database"
            dirList = ["data","model","payload","report"]
            for dir in dirList:
                dirs = root_path + "/" + dir
                if not os.path.exists(dirs):
                    os.mkdir(dirs)
        
            Output_column = payload['payloadData']['groundTruthClassLabel']

            dfs = []
            fields = []
            for filename in os.listdir(payload['report_path']):
                if filename.endswith('.csv'):
                    if filename == payload['modelName']+'.csv':   # reading original dataset
                        file_path = os.path.join(payload['report_path'], filename)
                        df = pd.read_csv(file_path)
                        df.drop(Output_column,axis=1,inplace=True)
                        df.insert(df.shape[1],"Attack",[0 for i in range(df.shape[0])])
                        for col in df.columns:
                            fields.append(col)
                        df = np.array(df)
                        dfs.append(df)
                    else:                                         # reading attack dataset
                        file_path = os.path.join(payload['report_path'], filename)
                        df = pd.read_csv(file_path)
                        df = df[df[df.columns[-1]] == True]
                        df.drop(Output_column,axis=1,inplace=True)
                        df.drop(columns=df.columns[-2:], axis=1, inplace=True)
                        cols = list(df.columns)
                        df.insert(df.shape[1],"Attack",[1 for i in range(df.shape[0])])
                        df = np.array(df)
                        dfs.append(df)       
            
            # creating defense_model path
            temp_path = root_path + "/cacheMemory"
            if(UT.isContentSafe({"Filename" : payload["modelName"]})):
                temp_path = os.path.join(temp_path,f'{payload["modelName"]}defenseModel.csv')
            else:
                log.info("Suspicious Data Involved in ModelName")
                return "Suspicious Data Involved in ModelName"

            if os.path.exists(temp_path):
                os.remove(temp_path)
            with open(temp_path,"w",newline="") as f:
                write = csv.writer(f)
                write.writerow(fields)
                for df in dfs:
                    write.writerows(df)
            
            # creating defense Model
            df = pd.read_csv(temp_path)
            X = df.loc[:,df.columns!="Attack"]
            Y = df["Attack"]
            X_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=1)
            model = XGBClassifier()
            model.fit(X_train,y_train)
            pickle.dump(model, open(os.path.join(payload['report_path'],"DefenseModel.pkl"), 'wb'))

            del model,X_train,x_test,y_train,y_test
            UT.databaseDelete(temp_path)

        except Exception as e:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "generateCombinedDenfenseModel", e, apiEndPoint, errorRequestMethod)
            raise Exception
        return "Success"


    def generateDenfenseModelendpoint(payload):
        
        try:
            payload_folder_path = UT.getcurrentDirectory() + "/database/payload"
            payload_path = ""
            SAFE_DIR = payload_folder_path
            def open_safe_file(filename):
                if '..' in filename or '/' in filename:
                    raise ValueError("Invalid filename")
                payload_path = os.path.join(SAFE_DIR, filename+".txt")
                return open(os.path.join(SAFE_DIR, filename+".txt"), "w", newline="")
            
            with open_safe_file(payload["modelName"]) as f:
                data = f.read()
            data = json.loads(data)
            Output_column = data["groundTruthClassLabel"]

            #reading original dataset
            data_path = UT.getcurrentDirectory()+f'/database/data/{payload["modelName"]}.csv'
            df1 = pd.read_csv(data_path, delimiter=',')
            df1.drop(Output_column,axis=1,inplace=True)
            df1.insert(df1.shape[1],"Attack",[0 for i in range(df1.shape[0])])

            root_path = UT.getcurrentDirectory() + "/database/report"
            report_folder = os.path.join(root_path,payload['folderName'])
            report_csv = os.path.join(report_folder,'Attack_Samples.csv')
            df2 = pd.read_csv(report_csv)
            df2.drop(Output_column,axis=1,inplace=True)
            df2.drop(columns=df2.columns[-2:], axis=1, inplace=True)
            cols = list(df2.columns)
            df2.insert(df2.shape[1],"Attack",[1 for i in range(df2.shape[0])])
    
            #creating defensemodel dataset
            fields=[]
            for col in df1.columns:
                fields.append(col)
            df1=np.array(df1)
            df2=np.array(df2)

            #creating defense_model path
            root_path = UT.getcurrentDirectory() + "/database"
            report_path = os.path.join(root_path+"/report",payload['folderName'])
            temp_path = root_path + "/cacheMemory"

            SAFE_DIR = temp_path
            def open_safe_file(filename):
                if '..' in filename or '/' in filename:
                    raise ValueError("Invalid filename")
                temp_path = os.path.join(SAFE_DIR, filename)
                return open(os.path.join(SAFE_DIR, filename),"w",newline="")
            with open_safe_file(f'{payload["modelName"]}defenseModel.csv') as f:
                write = csv.writer(f)
                write.writerow(fields)
                write.writerows(df1)
                write.writerows(df2)
        
            # creating defense Model
            df=pd.read_csv(temp_path)
            X=df.loc[:,df.columns!="Attack"]
            Y=df["Attack"]
            X_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1)
            model=XGBClassifier()
            model.fit(X_train,y_train)
            pickle.dump(model, open(os.path.join(report_path,"DefenseModel.pkl"), 'wb'))

            del model,X_train,x_test,y_train,y_test,df1,df2
            UT.databaseDelete(temp_path)
        except Exception as exc:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "generateDenfenseModelendpoint", exc, apiEndPoint, errorRequestMethod)
            raise Exception