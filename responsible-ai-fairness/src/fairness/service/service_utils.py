"""
Copyright 2024 Infosys Ltd.”

Use of this source code is governed by MIT license that can be found in the LICENSE file or at
MIT license https://opensource.org/licenses/MIT

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import json
from fastapi import HTTPException
import pandas
import matplotlib.pyplot as plt
import base64
import os
import time
import tempfile
import requests

from fairness.constants.local_constants import *
from fairness.dao.WorkBench.FileStoreDb import FileStoreReportDb
from fairness.dao.WorkBench.Filestore import FileStore
from nutanix_object_storage.nutanix_utility import NutanixObjectStorage


class Utils:
    def __init__(self) -> None:
        # self.fileStore = FileStoreReportDb()
        self.fileStore = FileStore()
    
    # def save_as_json_file(self,fileName:str,content):
    #     with open(fileName, "w") as outfile:
    #         json.dump(content,outfile,indent=2)
    
    def save_as_json_file(self, fileName:str, content1, content2):
        combined_content = {"content1": content1, "content2": content2}
        with open(fileName, "w") as outfile:
            json.dump(combined_content, outfile, indent=2)

    def save_as_file(self,filename:str, content):
        with open(filename,"wb") as outfile:
            outfile.write(content)
    
    def read_html_file(self,filename:str):
        with open(filename, 'r') as file:
            html_content = file.read()
        return html_content
    
    def json_to_html_individualMetric(self,json_obj2,label):
        log.info(json_obj2,"json_obj2")
        df = pandas.read_json(json.dumps(json_obj2))
        # Extract metric names and values
        data = df.iloc[0]
        content = data[label]
        log.info(content,"content")
        name = content['name']
        value = float(content['value'])
        desc=content['description']

        fig, ax = plt.subplots(figsize =(6,4))
        ax.bar(name, value, color='darkorchid')
 
        ax.set_xlabel("Individual metric")
        ax.set_ylabel("Value")
        ax.set_title(f"{name}")
        ax.set_ylim(-1, 1)
 
        # Save the plot as a PNG file
        path = "../output/"  # Replace with the desired path
        filename = os.path.join(path, f"{name.replace(' ', '_')}.png")
        plt.savefig(filename)
    
        # Convert the PNG file to base64
        with open(filename, "rb") as imagefile:
            encoded_image = base64.b64encode(imagefile.read()).decode()
 
            # Embed the base64 image in the HTML content
            #html_content = f"<h2 style='text-align:center; color:white; background-color:darkorchid; font-size:35px; font-family: sans-serif;'>Fairness Report</h2>"
        html_content = ""  
        html_content += f"<body><h3 style='color:darkorchid; text-align:left; font-size:21px'>{name}</h3></body>"
        html_content += f"<h3 style='font-weight:normal; font-family: sans-serif; font-size:17px;'>Description - {desc}</h3>"
        html_content += f"<h3 style='font-weight:normal; font-family: sans-serif; font-size:17px;'>Measured Value = {value}</h3>"
        html_content += f"<img src='data:image/png;base64,{encoded_image}' alt='{name} Plot'>"
            
        return html_content


    def json_to_html(self,json_obj,json_obj2,label):
        log.info(json_obj,"json_obj")
        log.info(json_obj2,"json_obj2")
        utils = Utils()
        utils.json_to_html_individualMetric(json_obj2,label)
        df = pandas.read_json(json.dumps(json_obj))
 
    # Extract metric names and values
        data = df.iloc[0]
        metrics = data['metrics']
 
        # Generate HTML content
        html_content = "<body><h2>FAIRNESS REPORT</h2>"
        html_content = f"<h2 style='text-align:center; color:white; background-color:darkorchid; font-size:35px; font-family: sans-serif;'>Fairness Report</h2>"
        html_content = html_content+f"<h3 style='font-weight:normal; font-family: sans-serif; font-size:17px;'>{F_Desc}</h3>"
        html_content = html_content+f"<h3 style='font-weight:normal; font-family: sans-serif; font-size:17px;'>{D_Desc}</h3>"
        html_content += f"<body><h3 style='color:darkorchid; text-align:left; font-size:23px'>METRICS</h3></body>"
 
    # Iterate over metrics and create plots
        for metric in metrics:
            metric_name = metric['name']
            metric_value = float(metric['value'])
            metric_desc=metric['description']
 
            # Create a new plot for each metric
            #plt.figure(figsize =(12,6))
            fig, ax = plt.subplots(figsize =(6,4))
            ax.bar(metric_name, metric_value, color='darkorchid')
 
            ax.set_xlabel("Metric")
            ax.set_ylabel("Value")
            ax.set_title(f"{metric_name}")
            ax.set_ylim(-1, 1)
 
            # Save the plot as a PNG file
            path = "../output/"  # Replace with the desired path
            filename = os.path.join(path, f"{metric_name.replace(' ', '_')}.png")
            plt.savefig(filename)
    
            # Convert the PNG file to base64
            with open(filename, "rb") as imagefile:
                encoded_image = base64.b64encode(imagefile.read()).decode()
 
            # Embed the base64 image in the HTML content
            #html_content = f"<h2 style='text-align:center; color:white; background-color:darkorchid; font-size:35px; font-family: sans-serif;'>Fairness Report</h2>"
            
            html_content += f"<body><h3 style='color:darkorchid; text-align:left; font-size:21px'>{metric_name}</h3></body>"
            html_content += f"<h3 style='font-weight:normal; font-family: sans-serif; font-size:17px;'>Description - {metric_desc}</h3>"
            html_content += f"<h3 style='font-weight:normal; font-family: sans-serif; font-size:17px;'>Measured Value = {metric_value}</h3>"
            html_content += f"<img src='data:image/png;base64,{encoded_image}' alt='{metric_name} Plot'>"
            
        result = utils.json_to_html_individualMetric(json_obj2,label)
        html_content += result
        html_content += "</body>"
        local_file_path = "../output/fairnessreport.html"
       
        return html_content
                
    def save_html_to_file(self,html_string, filename):
        with open(filename, 'w') as f:
            f.write(html_string)

    def parse_nutanix_bucket_object(self,fullpath: str):
        split_path = fullpath.split("//")
        return {'bucket_name': split_path[0], 'object_key': "/".join(split_path[1:])}
    
    def get_data_frame(self,extension: str,tpath: str,sep: str, usecols: list):
        if extension == "csv":
            return  pandas.read_csv(tpath,sep=",", usecols=usecols)
        elif extension=="parquet":
            return pandas.read_parquet(tpath,sep=",", usecols=usecols)
        elif extension == "feather":
            return pandas.read_feather(tpath,sep=",", usecols=usecols)
        elif extension == "json":
            return pandas.read_json(tpath,sep=",", usecols=usecols)
    
    def uploadfile_to_db(self,uploadPath,filePath):
         # to upload file in Nutanix
        buck_dict = self.parse_nutanix_bucket_object(uploadPath)
        bucket_ = buck_dict['bucket_name']
        key_ = buck_dict['object_key']
        strt_time = time.time()
        log.info("Start time", strt_time)
        fileName=os.path.basename(filePath)
        NutanixObjectStorage.upload_with_high_threshold(filePath, bucket_,
                                                        key_ + "/" + fileName, 10)
        end_time = time.time()
        log.info("End time", end_time)
        log.info("Total Time:", end_time - strt_time)
    
    def uploadfile_to_mongodb(self,filePath,fileType=None):
        log.info("filePath",filePath)
         # to upload file in Mongodb
        strt_time = time.time()
        log.info("Start time******", strt_time)
        fileId=self.fileStore.save_local_file(filePath=filePath,fileType=fileType)
        log.info(fileId)
        end_time = time.time()
        log.info("End time", end_time)
        log.info("Total Time:", end_time - strt_time)
        return fileId
    
    def modifyDf(self,df,catAttribute,labelmap,label):
        log.info(catAttribute)
        reverseLabelMap={v: k for k, v in labelmap.items()}
        def map_values(x):
            if int(x) == 1:
                return "privileged"
            else:
                return "unprivileged"
        for cat in catAttribute['name']:
            df[cat]=df[cat].apply(map_values).astype(str)
        df[label]=df[label].map(reverseLabelMap).astype(str)
        log.info("catAttribute", catAttribute)
        log.info("labelMap",labelmap)
        return df
    
    def pretrain_save_file(self,df: pandas.DataFrame, extension: str, file_path: str):
        log.info(df,"df************prttrain_save_file")
        log.info(extension,"extension************prttrain_save_file")
        log.info(file_path,"file_path************prttrain_save_file")
        if extension == "csv":
            df.to_csv(file_path, index=False)
        elif extension == "parquet":
            df.to_parquet(file_path, index=False)
        elif extension == "json":
            df.to_json(file_path, index=False, orient='records')
        elif extension == "feather":
            df.to_feather(file_path)

    def get_extension(self,fileName: str):
        if fileName.endswith(".csv"):
            return "csv"
        elif fileName.endswith(".feather"):
            return "feather"
        elif fileName.endswith(".parquet"):
            return "parquet"
        elif fileName.endswith(".json"):
            return "json"

    def get_data_frame(self,extension: str, fileName: str):
        # if extension == "csv":
        return pandas.read_csv(os.path.join(self.LOCAL_FILE_PATH, fileName))
        # elif extension=="parquet":
        #     return pandas.read_parquet(os.path.join(FairnessService.LOCAL_FILE_PATH, fileName))
        # elif extension == "feather":
        #     return pandas.read_feather(os.path.join(FairnessService.LOCAL_FILE_PATH, fileName))
        # elif extension == "json":
        #     return pandas.read_json(os.path.join(FairnessService.LOCAL_FILE_PATH, fileName))

    def store_file_locally(self,extension, file_content, file_path, file_name):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Create temporary file
            temp_file_name = temp_file.name
            # Write binary data to the temporary file
            file_content = file_content.read()
            temp_file.write(file_content)
            # Reset the file pointer to the beginning for reading
            temp_file.seek(0)

            df = pandas.DataFrame()
            if extension == "csv":
                df = pandas.read_csv(temp_file)
            elif extension == "parquet":
                df = pandas.read_parquet(temp_file)
            elif extension == "feather":
                df = pandas.read_feather(temp_file)
            elif extension == "json":
                df = pandas.read_json(temp_file)
            df.to_csv(os.path.join(file_path, file_name), index=False)
            # Close the temporary file before deletion
            temp_file.close()
        os.remove(temp_file_name)

    
    def store_file_locally_DB(self,extension, file_content, file_path, file_name):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Create temporary file
            temp_file_name = temp_file.name
            # Write binary data to the temporary file
            file_content = file_content
            temp_file.write(file_content)
            # Reset the file pointer to the beginning for reading
            temp_file.seek(0)

            df = pandas.DataFrame()
            if extension == "csv":
                df = pandas.read_csv(temp_file)
            elif extension == "parquet":
                df = pandas.read_parquet(temp_file)
            elif extension == "feather":
                df = pandas.read_feather(temp_file)
            elif extension == "json":
                df = pandas.read_json(temp_file)
            df.to_csv(os.path.join(file_path, file_name), index=False)
            # Close the temporary file before deletion
            temp_file.close()
        os.remove(temp_file_name)
        # with open(os.path.join(file_path, file_name), 'wb') as f:
        #     f.write(file_content.read())
        
        
    def parse_priv(self, priv):
        priv_list = []
        i = 0
        while (i < len(priv)):
            if "[" in priv[i]:
                i += 1
                priv_string = ""
                while (i < len(priv) and "]" not in priv[i]):
                    if "[" in priv[i]:
                        raise HTTPException(
                            status_code=400, detail="Priviledged attribute not closed properly")
                    priv_string += priv[i]
                    i += 1
                if "]" not in priv[i]:
                    raise HTTPException(
                        status_code=400, detail="Priviledged attribute not closed properly")
                i += 1
                log.info("Prev_string", priv_string)
                priv_array = priv_string.split(",")
                log.info(priv_array)
                priv_list.append(priv_array)
            else:
                if "," in priv[i]:
                    i += 1
                if "[" in priv[i]:
                    continue
                priv_string = ""
                while (i < len(priv) and "," not in priv[i]):
                    if priv[i] == '[' or priv[i] == ']':
                        raise HTTPException(
                            status_code=400, detail="Priviledged attribute not opened properly")
                    priv_string += priv[i]
                    i += 1
                if i < len(priv) and "," not in priv[i]:
                    raise HTTPException(
                        status_code=400, detail="Priviledged attribute not closed properly")
                priv_list.append([priv_string])
                i += 1
        return priv_list
    

    # def html_to_pdf(self,batchId):
    #     # url = os.getenv("REPORT_URL")
    #     # url = "http://10.212.115.38:30105/v1/report/htmltopdfconversion"
    #     url = "http://localhost/v1/report/htmltopdfconversion"
    #     log.info("excuted*********************")
    #     payload = {"batchId": batchId}

    #     response = requests.request(
    #         "POST", url, data=payload, verify=False).json()
    #     log.info("sucessfully executed the request")
    #     log.info(response)