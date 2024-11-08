'''
MIT license https://opensource.org/licenses/MIT
Copyright 2024 Infosys Ltd
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''




import os
import shutil
import pandas as pd
import numpy as np
import csv
import json
import matplotlib.pyplot as plt

from app.config.urls import UrlLinks as UL
from app.service.utility import Utility as UT
from app.service.defence import Defence as DF
from app.config.logger import CustomLogger
import concurrent.futures as con

telemetry_flg =os.getenv("TELEMETRY_FLAG")

apiEndPoint ='/v1/security/model'
errorRequestMethod = 'GET'

log =CustomLogger()

class Report:

    def generateinferencereport(payload):

        try:
            root_path = os.getcwd()
            root_path = UT.getcurrentDirectory() + "/database"
            dirList = ["data","model","payload","report"]
            for dir in dirList:
                dirs = root_path + "/" + dir
                if not os.path.exists(dirs):
                    os.mkdir(dirs)

            Current_Report_ID = UL.Current_ID + 1
            foldername = f'{payload["attackName"]}_{Current_Report_ID}'
            xlfilename = 'Attack_Samples.xlsx'
            root_path = root_path + "/report"
            report_path = os.path.join(root_path,foldername)
            if os.path.isdir(report_path):    #These 2Lines are anyway useless with unique ART_ID
                return foldername
            os.mkdir(report_path)

            payload_folder_path = UT.getcurrentDirectory() + "/database/payload"
            payload_path = os.path.join(payload_folder_path,payload['modelName'] + ".txt")
            with open(f'{payload_path}') as f:
                data = f.read()
            data = json.loads(data)
            type = data['targetClassifier']
            tag = data['targetDataType']


            field_names = ['Name', 'Type', 'Category', 'Tag', 'FrameWork', 'Docs']
            dict={"Name":payload["attackName"],"Type":type,"Category":'Inference',"Tag":tag,'FrameWork':'art'}
            # if payload["attackName"] == "AttributeInference":
            #     dict['Docs'] = "There are implementations available for Attribute Inference Black Box Attack in Machine Learning Algorithms. | End-TO-End Implementation Paper link: https://arxiv.org/abs/2012.03404 "
            # elif payload["attackName"] == "LabelOnlyDecisionBoundary":
            #     dict['Docs'] = "Implementation of the algorithms described in the paper Membership Leakage in Label-Only Exposures by Zheng Li,  Yang Zhang | Attack paper link : https://arxiv.org/abs/2007.15528"
            if payload["attackName"] == "LabelOnlyDecisionBoundaryAttackEndPoint":
                dict['Docs'] = "Implementation of the algorithms described in the paper Membership Leakage in Label-Only Exposures by Zheng Li,  Yang Zhang | Attack paper link : https://arxiv.org/abs/2007.15528"
            # elif payload["attackName"] == "AttributeInferenceWhiteBoxLifestyleDecisionTree":
            #     dict['Docs'] = "There are implementations available for Attribute Inference Black Box Attack in Machine Learning Algorithms. | End-TO-End Implementation Paper link: https://arxiv.org/abs/2012.03404 "
            # elif payload["attackName"] == "AttributeInferenceWhiteBoxDecisionTree":
            #     dict['Docs'] = "There are implementations available for Attribute Inference Black Box Attack in Machine Learning Algorithms. | End-TO-End Implementation Paper link: https://arxiv.org/pdf/2007.14321 "
            # elif payload["attackName"] == "InferenceLabelOnlyGap":
            #     dict['Docs'] = "There are implementations available for Attribute Inference Black Box Attack in Machine Learning Algorithms. | End-TO-End Implementation Paper link: https://arxiv.org/abs/2012.03404 "
            elif payload["attackName"] == "LabelOnlyGapAttackEndPoint":
                dict['Docs'] = "There are implementations available for Attribute Inference Black Box Attack in Machine Learning Algorithms. | End-TO-End Implementation Paper link: https://arxiv.org/abs/2012.03404 "
            # elif payload["attackName"] == "MembershipInferenceRule":
            #     dict['Docs'] = "Implementation of the algorithms described in the paper Membership Inference Attacks against Machine Learning Models by Reza Shokri, Marco Stronati. | Attack paper link : https://arxiv.org/abs/1610.05820"
            elif payload["attackName"] == "MembershipInferenceBlackBoxRuleBasedAttackEndPoint":
                dict['Docs'] = "Implementation of the algorithms described in the paper Membership Inference Attacks against Machine Learning Models by Reza Shokri, Marco Stronati. | Attack paper link : https://arxiv.org/abs/1610.05820"
            # elif payload["attackName"] == "MembershipInferenceBlackBox":
            #     dict['Docs'] = "Implementation of the algorithms described in the paper Membership Inference Attacks against Machine Learning Models by Reza Shokri, Marco Stronati. | Attack paper link : https://arxiv.org/abs/1610.05820"
            elif payload["attackName"] == "MembershipInferenceBlackBoxAttackEndPoint":
                dict['Docs'] = "Implementation of the algorithms described in the paper Membership Inference Attacks against Machine Learning Models by Reza Shokri, Marco Stronati. | Attack paper link : https://arxiv.org/abs/1610.05820"
            # elif payload["attackName"] == "Poisoning":
            #     dict['Docs'] = "Implementation of the algorithms described in the paper Poisoning Attacks against Support Vector Machines by Battista Biggio,  Blaine Nelson. | Attack paper link : https://arxiv.org/abs/1206.6389" 
            table1 = [dict]
            junk_folder = UT.getcurrentDirectory() + f"/database/cacheMemory"
            csv_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.csv')
            html_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.html')
            if os.path.exists(csv_junk):
                os.remove(csv_junk)
            if os.path.exists(html_junk):
                os.remove(html_junk)
            with open(csv_junk, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = field_names)
                writer.writeheader()
                writer.writerows(table1)
            a=pd.read_csv(csv_junk)
            a.to_html(os.path.join(report_path,html_junk))
            with open(os.path.join(report_path,html_junk),"r") as file1:
                data1=file1.readlines()   


            field_names = ['Attack id', 'Target Name', 'Attack Name', 'Status', 'Success']
            dict={'Attack id':f'InfosysModelReport{Current_Report_ID}','Target Name':payload['modelName'],'Attack Name':payload["attackName"],'Status':'complete','Success':"{'default': True, 'current': True, 'previous': True, 'docs': True}"}
            table2 = [dict]
            junk_folder = UT.getcurrentDirectory() + f"/database/cacheMemory"
            csv_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.csv')
            html_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.html')
            if os.path.exists(csv_junk):
                os.remove(csv_junk)
            if os.path.exists(html_junk):
                os.remove(html_junk)
            with open(csv_junk, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = field_names)
                writer.writeheader()
                writer.writerows(table2)
            a=pd.read_csv(csv_junk)
            a.to_html(os.path.join(report_path,html_junk))
            with open(os.path.join(report_path,html_junk),"r") as file2:
                data2=file2.readlines()


            if payload["attackName"] in ['AttributeInference', 'InferenceLabelOnlyGap', 'LabelOnlyGapAttackEndPoint', 'MembershipInferenceBlackBoxRuleBasedAttackEndPoint', 'LabelOnlyDecisionBoundaryAttackEndPoint', 'MembershipInferenceBlackBoxAttackEndPoint']: 
                field_names = ['index','Inference Attack Accuracy','success']      
                dict={"index":'0',"Inference Attack Accuracy":payload['inference_acc']}
                if payload['inference_acc'] >= 0.5:
                    dict['success']='True'
                else:
                    dict['success']='False'           
            # elif payload["attackName"] == 'LabelOnlyDecisionBoundary':
            #     field_names = ['index','Inference Attack Accuracy','Infered Data Value','success']      
            #     dict={"index":'0',"Inference Attack Accuracy":payload['inference_acc'],"Infered Data Value":payload['infered_data']}
            #     if payload['inference_acc'] >= 0.5:
            #         dict['success']='True'
            #     else:
            #         dict['success']='False' 
            # elif payload["attackName"] in ['AttributeInferenceWhiteBoxDecisionTree','AttributeInferenceWhiteBoxLifestyleDecisionTree']:
            #     field_names = ['index','Inference Attack Accuracy','Inferred Features','success']      
            #     dict={"index":'0',"Inference Attack Accuracy":payload['inference_acc'],"Inferred Features":payload['inference_feature']}
            #     if payload['inference_acc'] >= 0.5:
            #         dict['success']='True'
            #     else:
            #         dict['success']='False' 
            # elif payload["attackName"] in ['MembershipInferenceBlackBox','MembershipInferenceRule']:
            #     field_names = ['index','Inference Attack Accuracy','Precision','Recall','success']      
            #     dict={"index":'0',"Inference Attack Accuracy":payload['inference_acc'],"Precision":payload['precision'],"Recall":payload['recall']}
            #     if payload['inference_acc'] >= 0.5:
            #         dict['success']='True'
            #     else:
            #         dict['success']='False' 
            # elif payload["attackName"] == "Poisoning":
            #     field_names = ['index','Clean Model Train Accuracy','Clean Model Test Accuracy']      
            #     dict={"index":'0',"Clean Model Train Accuracy":payload['clean_train_acc'],"Clean Model Test Accuracy":payload['clean_test_acc']}  
            table3 = [dict]
            junk_folder = UT.getcurrentDirectory() + f"/database/cacheMemory"
            csv_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.csv')
            html_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.html')
            if os.path.exists(csv_junk):
                os.remove(csv_junk)
            if os.path.exists(html_junk):
                os.remove(html_junk)
            with open(csv_junk, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = field_names)
                writer.writeheader()
                writer.writerows(table3)
            a=pd.read_csv(csv_junk)
            a.to_html(os.path.join(report_path,html_junk))
            with open(os.path.join(report_path,html_junk),"r") as file3:
                data3=file3.readlines()
            

            # if payload["attackName"] == "Poisoning":
            #     field_names = ['index','Poison Model Train Accuracy','Poison Model Test Accuracy','Success']      
            #     dict={"index":'0',"Poison Model Train Accuracy":payload['poison_train_acc'],"Poison Model Test Accuracy":payload['poison_test_acc']}
            #     if (payload['poison_train_acc'] >= 0.5 and payload['poison_test_acc'] >= 0.5):
            #         dict['Success'] = 'True'
            #     else:
            #         dict['Success'] = 'Failed'  
            #     table4 = [dict]
            #     junk_folder = UT.getcurrentDirectory() + f"/database/cacheMemory"
            #     csv_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.csv')
            #     html_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.html')
            #     if os.path.exists(csv_junk):
            #         os.remove(csv_junk)
            #     if os.path.exists(html_junk):
            #         os.remove(html_junk)
            #     with open(csv_junk, 'w') as csvfile:
            #         writer = csv.DictWriter(csvfile, fieldnames = field_names)
            #         writer.writeheader()
            #         writer.writerows(table4)
            #     a=pd.read_csv(csv_junk)
            #     a.to_html(os.path.join(report_path,html_junk))
            #     with open(os.path.join(report_path,html_junk),"r") as file4:
            #         data4=file4.readlines()
  
            #     with open(os.path.join(report_path,f"report.html"),"w") as file:
            #         file.writelines(UT.htmlCssContent())
            #         file.writelines('<body>')
            #         file.writelines(f'<h2>{payload["attackName"]}_Attack</h2>')
            #         file.writelines(UT.htmlContent(payload["attackName"]))
            #         file.writelines("<h3>Selected Attack</h3>")
            #         file.writelines(data1)
            #         file.writelines("<h3>Attack Status</h3>")
            #         file.writelines(data2)
            #         file.writelines("<h3>Clean Model Accuracy</h3>")
            #         file.writelines(data3)
            #         file.writelines("<h3>Accuracy After Attack</h3>")
            #         file.writelines(data4)
            #         # file.write("<h3 style='text-align:center;'>Copyright \u00A9 2023 Infosys Limited | Internal Communications - HRD</h3>")
            #         file.writelines('</body>')
            # else:
            #     with open(os.path.join(report_path,f"report.html"),"w") as file:
            #         file.writelines(UT.htmlCssContent())
            #         file.writelines('<body>')
            #         file.writelines(f'<h2>{payload["attackName"]}_Attack</h2>')
            #         file.writelines(UT.htmlContent(payload["attackName"]))
            #         file.writelines("<h3>Selected Attack</h3>")
            #         file.writelines(data1)
            #         file.writelines("<h3>Attack Status</h3>")
            #         file.writelines(data2)
            #         file.writelines("<h3>Inference Attack Output</h3>")
            #         file.writelines(data3)   
            #         file.writelines('</body>')
            with open(os.path.join(report_path,f"report.html"),"w") as file:
                file.writelines(UT.htmlCssContent())
                file.writelines('<body>')
                file.writelines(f'<h2>{payload["attackName"]}_Attack</h2>')
                file.writelines(UT.htmlContent(payload["attackName"]))
                file.writelines("<h3>Selected Attack</h3>")
                file.writelines(data1)
                file.writelines("<h3>Attack Status</h3>")
                file.writelines(data2)
                file.writelines("<h3>Inference Attack Output</h3>")
                file.writelines(data3)   
                file.writelines('</body>')
            # UT.htmlToPdfWithWatermark({'folder_path':report_path})

            shutil.make_archive(report_path,'zip',report_path)
            UT.updateCurrentID()
            UT.databaseDelete(csv_junk)
            UT.databaseDelete(html_junk)
            return foldername
        
        except Exception as exc:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "generateinferencereport", exc, apiEndPoint, errorRequestMethod)
            raise Exception


    # def generateimagereport1(payload):
        
    #     try:
    #         root_path = os.getcwd()
    #         root_path = UT.getcurrentDirectory() + "/database"
    #         dirList = ["data","model","payload","report"]
    #         for dir in dirList:
    #             dirs = root_path + "/" + dir
    #             if not os.path.exists(dirs):
    #                 os.mkdir(dirs)

    #         Current_Report_ID = UL.Current_ID + 1
    #         foldername = f'{payload["attackName"]}_{Current_Report_ID}'
    #         xlfilename = 'Attack_Samples.xlsx'
    #         root_path = root_path + "/report"
    #         report_path = os.path.join(root_path,foldername)
    #         if os.path.isdir(report_path):    #These 2Lines are anyway useless with unique ART_ID
    #             return foldername
    #         os.mkdir(report_path)


    #         field_names = [f'Base Model Name','Defect Class Name','Sample Data','Confidence Score']
    #         dict={f'Base Model Name':payload["modelName"],'Defect Class Name':payload['basePrediction_class'],'Sample Data':payload['base_sample'],'Confidence Score':payload['baseActual_confidence']}
    #         table1 = [dict]
    #         junk_folder = UT.getcurrentDirectory() + f"/database/cacheMemory"
    #         csv_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.csv')
    #         html_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.html')
    #         if os.path.exists(csv_junk):
    #             os.remove(csv_junk)
    #         if os.path.exists(html_junk):
    #             os.remove(html_junk)
    #         with open(csv_junk, 'w') as csvfile:
    #             writer = csv.DictWriter(csvfile, fieldnames = field_names)
    #             writer.writeheader()
    #             writer.writerows(table1)
    #         a=pd.read_csv(csv_junk)
    #         a.to_html(os.path.join(report_path,html_junk))
    #         with open(os.path.join(report_path,html_junk),"r") as file1:
    #             data1=file1.readlines()


    #         field_names = [f'Attack Name','Defect Class Name','Adversarial Sample Data','Confidence Score', 'Success']
    #         val:any
    #         if payload['basePrediction_class'] == payload['adversialPrediction_class']:
    #             val = 'False'
    #         else:
    #             val = 'True'
    #         dict={f'Attack Name':payload["attackName"],'Defect Class Name':payload['adversialPrediction_class'],'Adversarial Sample Data':payload['adversial_sample'],'Confidence Score':payload['adversialActual_confidence'], 'Success':val}
    #         table2 = [dict]
    #         junk_folder = UT.getcurrentDirectory() + f"/database/cacheMemory"
    #         csv_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.csv')
    #         html_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.html')
    #         if os.path.exists(csv_junk):
    #             os.remove(csv_junk)
    #         if os.path.exists(html_junk):
    #             os.remove(html_junk)
    #         with open(csv_junk, 'w') as csvfile:
    #             writer = csv.DictWriter(csvfile, fieldnames = field_names)
    #             writer.writeheader()
    #             writer.writerows(table2)
    #         a=pd.read_csv(csv_junk)
    #         a.to_html(os.path.join(report_path,html_junk))
    #         with open(os.path.join(report_path,html_junk),"r") as file2:
    #             data2=file2.readlines()


    #         with open(os.path.join(report_path,f"report.html"),"w") as file:
    #             file.writelines(UT.htmlCssContent())
    #             file.writelines('<body>')
    #             file.writelines(f'<h2>{payload["attackName"]}_Attack</h2>')
    #             file.writelines(UT.htmlContent(payload["attackName"]))
    #             file.writelines("<h3>Actual Result</h3>")
    #             file.writelines(data1)
    #             file.writelines("<h3>Result After Attack</h3>")
    #             file.writelines(data2)
    #             # file.write("<h3 style='text-align:center;'>Copyright \u00A9 2023 Infosys Limited | Internal Communications - HRD</h3>")
    #             file.writelines('</body>')

    #         # UT.htmlToPdfWithWatermark({'folder_path':report_path})

    #         shutil.make_archive(report_path,'zip',report_path)
    #         UT.updateCurrentID()
    #         UT.databaseDelete(csv_junk)
    #         UT.databaseDelete(html_junk)
    #         return foldername
        
    #     except Exception as exc:
    #         print(exc)
    #         raise Exception


    def generateimagereport(payload):
        
        try:
            root_path = os.getcwd()
            root_path = UT.getcurrentDirectory() + "/database"
            dirList = ["data","model","payload","report"]
            for dir in dirList:
                dirs = root_path + "/" + dir
                if not os.path.exists(dirs):
                    os.mkdir(dirs)

            Current_Report_ID = UL.Current_ID + 1
            foldername = f'{payload["attackName"]}_{Current_Report_ID}'
            xlfilename = 'Attack_Samples.xlsx'
            root_path = root_path + "/report"
            report_path = os.path.join(root_path,foldername)
            if os.path.isdir(report_path):    #These 2Lines are anyway useless with unique ART_ID
                return foldername
            os.mkdir(report_path)

            # Generating Attack Samples
            # UT.generateImage({'base_sample':payload['base_sample'],'adversial_sample':payload['adversial_sample'],'attackName':payload['attackName'],'report_path':report_path})
            img_path = os.path.join(report_path, f"{payload['imageName']}.png")
            plt.imshow(payload['adversial_sample'][0])
            plt.axis('off')
            plt.title('Adversarial Sample')
            plt.savefig(img_path)
            plt.close()

            # field_names = [f'Base Model Name','Defect Class Name','Confidence Score']
            # dict={f'Base Model Name':payload["modelName"],'Defect Class Name':payload['basePrediction_class'],'Confidence Score':payload['baseActual_confidence']}
            field_names = [f'Base Model Name','Actual Labels','Confidence Score']
            dict={f'Base Model Name':payload["modelName"],'Actual Labels':payload['basePrediction_class'],'Confidence Score':payload['baseActual_confidence']}
            table1 = [dict]
            junk_folder = UT.getcurrentDirectory() + f"/database/cacheMemory"
            csv_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.csv')
            html_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.html')
            if os.path.exists(csv_junk):
                os.remove(csv_junk)
            if os.path.exists(html_junk):
                os.remove(html_junk)
            with open(csv_junk, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = field_names)
                writer.writeheader()
                writer.writerows(table1)
            a=pd.read_csv(csv_junk)
            a.to_html(os.path.join(report_path,html_junk))
            with open(os.path.join(report_path,html_junk),"r") as file1:
                data1=file1.readlines()


            # field_names = [f'Attack Name','Defect Class Name','Confidence Score', 'Success']
            field_names = [f'Image Name', 'Actual Labels', 'Final Labels','Confidence Score', 'Success']
            val:any
            if payload['basePrediction_class'] == payload['adversialPrediction_class']:
                val = 'False'
            else:
                val = 'True'
            # dict={f'Attack Name':payload["attackName"],'Defect Class Name':payload['adversialPrediction_class'],'Confidence Score':payload['adversialActual_confidence'], 'Success':val}
            dict={f'Image Name':payload["imageName"], 'Actual Labels':payload['basePrediction_class'], 'Final Labels':payload['adversialPrediction_class'], 'Confidence Score':payload['adversialActual_confidence'], 'Success':val}
            table2 = [dict]
            junk_folder = UT.getcurrentDirectory() + f"/database/cacheMemory"
            csv_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.csv')
            html_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.html')
            if os.path.exists(csv_junk):
                os.remove(csv_junk)
            if os.path.exists(html_junk):
                os.remove(html_junk)
            with open(csv_junk, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = field_names)
                writer.writeheader()
                writer.writerows(table2)
            a=pd.read_csv(csv_junk)
            a.to_html(os.path.join(report_path,html_junk))
            with open(os.path.join(report_path,html_junk),"r") as file2:
                data2=file2.readlines()


            with open(os.path.join(report_path,f"report.html"),"w") as file:
                file.writelines(UT.htmlCssContent())
                file.writelines('<body>')
                file.writelines(f'<h2>{payload["attackName"]}_Attack</h2>')
                file.writelines(UT.htmlContent(payload["attackName"]))
                file.writelines("<h3>Actual Result</h3>")
                file.writelines(data1)
                file.writelines("<h3>Result After Attack</h3>")
                file.writelines(data2)
                # file.write("<h3 style='text-align:center;'>Copyright \u00A9 2023 Infosys Limited | Internal Communications - HRD</h3>")
                file.writelines('</body>')

            # UT.htmlToPdfWithWatermark({'folder_path':report_path})

            shutil.make_archive(report_path,'zip',report_path)
            UT.updateCurrentID()
            UT.databaseDelete(csv_junk)
            UT.databaseDelete(html_junk)
            return foldername
        
        except Exception as exc:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "generateimagereport", exc, apiEndPoint, errorRequestMethod)
            raise Exception

    
    def generatecsvreportart(payload):
        
        try:
            root_path = os.getcwd()
            root_path = UT.getcurrentDirectory() + "/database"
            dirList = ["data","model","payload","report"]
            for dir in dirList:
                dirs = root_path + "/" + dir
                if not os.path.exists(dirs):
                    os.mkdir(dirs)

            Current_Report_ID = UL.Current_ID + 1
            foldername = f'{payload["attackName"]}_{Current_Report_ID}'
            csvfilename = 'Attack_Samples.csv'
            root_path = root_path + "/report"
            report_path = os.path.join(root_path,foldername)
            if os.path.isdir(report_path):    #These 2Lines are anyway useless with unique ART_ID
                return foldername
            os.mkdir(report_path)

            payload_folder_path = UT.getcurrentDirectory() + "/database/payload"
            payload_path = os.path.join(payload_folder_path,payload['modelName'] + ".txt")
            with open(f'{payload_path}') as f:
                data = f.read()
            data = json.loads(data)
            type = data['targetClassifier']
            tag = data['targetDataType']

            # Generating Attack Samples
            if payload["attackName"] == "HopSkipJumpImage":
                with open(os.path.join(report_path,csvfilename), 'w',newline="") as f:
                    write = csv.writer(f)
                    write.writerow(payload['columns'])
                    write.writerows(payload['attack_array'])
            else:
                with open(os.path.join(report_path,csvfilename), 'w',newline="") as f:
                    write = csv.writer(f)
                    write.writerow(payload['columns'])
                    write.writerows(payload['adversial_sample'])

            #Generating Defense Model
            DF.generateDenfenseModel({'modelName':payload['modelName'],'folderName':foldername, 'dataFileName':payload['dataFileName']})


            field_names = ['Attack id', 'Model Name', 'Attack Name', 'Status', 'Mean Difference']
            dict={'Attack id':f'InfosysModelReport{Current_Report_ID}','Model Name':payload['modelName'],'Attack Name':payload["attackName"],'Status':'Complete','Mean Difference':payload['perturbation']}
            table3 = [dict]
            junk_folder = UT.getcurrentDirectory() + f"/database/cacheMemory"
            csv_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.csv')
            html_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.html')
            if os.path.exists(csv_junk):
                os.remove(csv_junk)
            if os.path.exists(html_junk):
                os.remove(html_junk)
            with open(csv_junk, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = field_names)
                writer.writeheader()
                writer.writerows(table3)
            a=pd.read_csv(csv_junk)
            a.to_html(os.path.join(report_path,html_junk))
            with open(os.path.join(report_path,html_junk),"r") as file3:
                data3=file3.readlines()

            junk_folder = UT.getcurrentDirectory() + f"/database/cacheMemory"
            csv_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.csv')
            html_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.html')
            if os.path.exists(csv_junk):
                os.remove(csv_junk)
            if os.path.exists(html_junk):
                os.remove(html_junk)

            field_names = ['sample_index','actual_labels','final_labels','success']
            dict = np.array(payload['attack_data_status'])
            with open(csv_junk,"w",newline="") as f:
                write = csv.writer(f)
                write.writerow(field_names)
                write.writerows(dict)
            a=pd.read_csv(csv_junk)
            a.to_html(os.path.join(report_path,html_junk))
            with open(os.path.join(report_path,html_junk),"r") as file4:
                data4=file4.readlines()

            # generating graph for particular attack  
            csv_path = os.path.join(report_path,csvfilename)
            df = pd.read_csv(csv_path)
            cols = list(df.columns)
            key = 0
            if data['groundTruthClassLabel'] in cols and 'prediction' in cols:
                graph_html = UT.graphForAttack({'folder_path':report_path, 'target':data['groundTruthClassLabel'], 'attackName':payload["attackName"]})
                key = 1
            
            with open(os.path.join(report_path,f"report.html"),"w") as file:
                file.writelines(UT.htmlCssContent())
                file.writelines('<body>')
                file.writelines(f'<h2>{payload["attackName"]}_Attack</h2>')
                file.writelines(UT.htmlContent(payload["attackName"]))
                if key == 1:
                    file.writelines(f"<h3>Visualisation for Prediction Data</h3>")
                    file.writelines(graph_html)
                file.writelines("<h3>Attack Status</h3>")
                file.writelines(data3)
                if len(payload['attack_data_status']) != 0:
                    file.writelines("<h3>Adversial Input used for attack and Adversial output generated</h3>")
                    file.writelines(data4)
                # file.write("<h3 style='text-align:center;'>Copyright \u00A9 2023 Infosys Limited | Internal Communications - HRD</h3>")
                file.writelines('</body>')
            
            # UT.htmlToPdfWithWatermark({'folder_path':report_path})
            
            shutil.make_archive(report_path,'zip',report_path)
            UT.updateCurrentID()
            UT.databaseDelete(csv_junk)
            UT.databaseDelete(html_junk)
            return foldername
        
        except Exception as exc:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "generatecsvreportart", exc, apiEndPoint, errorRequestMethod)
            raise Exception
        

    def generatecsvreportart1(payload):

        try:
            root_path = os.getcwd()
            root_path = UT.getcurrentDirectory() + "/database"
            dirList = ["data","model","payload","report"]
            for dir in dirList:
                dirs = root_path + "/" + dir
                if not os.path.exists(dirs):
                    os.mkdir(dirs)

            Current_Report_ID = UL.Current_ID + 1
            foldername = f'{payload["attackName"]}_{Current_Report_ID}'
            csvfilename = 'Attack_Samples.csv'
            root_path = root_path + "/report"
            report_path = os.path.join(root_path,foldername)
            if os.path.isdir(report_path):    #These 2Lines are anyway useless with unique ART_ID
                return foldername
            os.mkdir(report_path)

            payload_folder_path = UT.getcurrentDirectory() + "/database/payload"
            payload_path = os.path.join(payload_folder_path,payload['modelName'] + ".txt")
            with open(f'{payload_path}') as f:
                data = f.read()
            data = json.loads(data)
            type = data['targetClassifier']
            tag = data['targetDataType']

            # Generating Attack Samples
            if payload["attackName"] == "HopSkipJumpImage":
                with open(os.path.join(report_path,csvfilename), 'w',newline="") as f:
                    write = csv.writer(f)
                    write.writerow(payload['columns'])
                    write.writerows(payload['attack_array'])
            else:
                with open(os.path.join(report_path,csvfilename), 'w',newline="") as f:
                    write = csv.writer(f)
                    write.writerow(payload['columns'])
                    write.writerows(payload['adversial_sample'])

            #Generating Defense Model
            DF.generateDenfenseModel({'modelName':payload['modelName'],'folderName':foldername, 'dataFileName':payload['dataFileName']})

            # generating graph for particular attack  
            csv_path = os.path.join(report_path,csvfilename)
            df = pd.read_csv(csv_path)
            cols = list(df.columns)
            key = 0
            # if 'target' in cols and 'prediction' in cols:
            #     graph_html = UT.graphForAttack({'folder_path':report_path})
            #     key = 1
            if data['groundTruthClassLabel'] in cols and 'prediction' in cols:
                graph_html = UT.graphForAttack({'folder_path':report_path, 'target':data['groundTruthClassLabel'], 'attackName':payload["attackName"]})
                key = 1
            
            # Attack Status Data
            field_names = ['Attack id', 'Model Name', 'Attack Name', 'Status', 'Mean Difference']
            attack_status_dict={'Attack id':f'InfosysModelReport{Current_Report_ID}','Model Name':payload['modelName'],'Attack Name':payload["attackName"],'Status':'Complete','Mean Difference':payload['perturbation']}
            # attack_status_row = f"""
            #     <tr>
            #         <td>{attack_status_dict['Attack id']}</td>
            #         <td>{attack_status_dict['Model Name']}</td>
            #         <td>{attack_status_dict['Attack Name']}</td>
            #         <td>{attack_status_dict['Status']}</td>
            #         <td>{attack_status_dict['Mean Difference']:.4f}</td>
            #     </tr>
            # """
            attack_status_row = f"""
                <tr>
                    <td>
                        <div class="attack-data-table-column">
                            <div class="attack-data-table-column-value">{attack_status_dict['Attack id']}</div>
                        </div>
                    </td>
                    <td>
                        <div class="attack-data-table-column">
                            <div class="attack-data-table-column-value">{attack_status_dict['Model Name']}</div>
                        </div>
                    </td>
                    <td>
                        <div class="attack-data-table-column">
                            <div class="attack-data-table-column-value">{attack_status_dict['Attack Name']}</div>
                        </div>
                    </td>
                    <td>
                        <div class="attack-data-table-column">
                            <div class="attack-data-table-column-value">{attack_status_dict['Status']}</div>
                        </div>
                    </td>
                    <td>
                        <div class="attack-data-table-column">
                            <div class="attack-data-table-column-value">{attack_status_dict['Mean Difference']:.4f}</div>
                        </div>
                    </td>
                </tr>
            """

            # Attack Input-Output Data
            attack_input_output_list = []
            for i in range(len(payload['attack_data_status'])):
                d = {}
                d['sample_index'] = payload['attack_data_status'][i][0]
                d['actual_labels'] = payload['attack_data_status'][i][1]
                d['final_labels'] = payload['attack_data_status'][i][2]
                d['success'] = payload['attack_data_status'][i][3]
                attack_input_output_list.append(d)
            attack_ipop_row = ""
            for data_list in attack_input_output_list:
                row = f"""
                <tr>
                    <td>{data_list['sample_index']}</td>
                    <td>{data_list['actual_labels']}</td>
                    <td>{data_list['final_labels']}</td>
                    <td>{data_list['success']}</td>
                </tr>
                """
                attack_ipop_row += row
            
            # Call htmlContentReport
            html_data = UT.htmlContentReport({'attackName':payload["attackName"],'graph_html':graph_html,'attack_status_row':attack_status_row,'attack_ipop_row':attack_ipop_row})

            with open(os.path.join(report_path,f"report.html"),"w") as file:
                file.writelines(UT.htmlCssContentReport())
                file.writelines(html_data)

            # UT.htmlToPdfWithWatermark({'folder_path':report_path})
            
            shutil.make_archive(report_path,'zip',report_path)
            UT.updateCurrentID()
            return foldername
        
        except Exception as exc:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "generatecsvreportart1", exc, apiEndPoint, errorRequestMethod)
            raise Exception


    def generatecsvreportartendpoint(payload):
        
        try:
            root_path = os.getcwd()
            root_path = UT.getcurrentDirectory() + "/database"
            dirList = ["data","model","payload","report"]
            for dir in dirList:
                dirs = root_path + "/" + dir
                if not os.path.exists(dirs):
                    os.mkdir(dirs)
            Current_Report_ID = UL.Current_ID + 1 
            foldername = f'{payload["attackName"]}_{Current_Report_ID}'
            csvfilename = 'Attack_Samples.csv'
            root_path = root_path + "/report"
            report_path = os.path.join(root_path,foldername)
            if os.path.isdir(report_path):    #These 2Lines are anyway useless with unique ART_ID
                return foldername 
            os.mkdir(report_path)
            # payload_folder_path = os.getcwd()[:-4] + "/database/payload"
            payload_folder_path = UT.getcurrentDirectory() + "/database/payload"
            payload_path = os.path.join(payload_folder_path,payload['modelName'] + ".txt")
            with open(f'{payload_path}') as f:
                data = f.read()
            data = json.loads(data)
            # type = data['targetClassifier']
            # tag = data['targetDataType']
            # Generating Attack Samples
            if payload["attackName"] == "HopSkipJumpImage":
                with open(os.path.join(report_path,csvfilename), 'w',newline="") as f:
                    write = csv.writer(f)
                    write.writerow(payload['columns'])
                    write.writerows(payload['attack_array'])
            else:
                with open(os.path.join(report_path,csvfilename), 'w',newline="") as f:
                    write = csv.writer(f)
                    write.writerow(payload['columns'])
                    write.writerows(payload['adversial_sample'])
            #Generating Defense Model
            DF.generateDenfenseModelendpoint({'modelName':payload['modelName'],'folderName':foldername})
            field_names = ['Attack id', 'Model Name', 'Attack Name', 'Status', 'Mean Difference']
            dict={'Attack id':f'InfosysModelReport{Current_Report_ID}','Model Name':payload['modelName'],'Attack Name':payload["attackName"],'Status':'Complete','Mean Difference':payload['perturbation']}
            table3 = [dict]
            # junk_folder = os.getcwd()[:-4] + f"/database/cacheMemory"
            junk_folder = UT.getcurrentDirectory() + f"/database/cacheMemory"
            csv_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.csv')
            html_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.html')
            if os.path.exists(csv_junk):
                os.remove(csv_junk)
            if os.path.exists(html_junk):
                os.remove(html_junk)
            with open(csv_junk, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = field_names)
                writer.writeheader()
                writer.writerows(table3)
            a=pd.read_csv(csv_junk)
            a.to_html(os.path.join(report_path,html_junk))
            with open(os.path.join(report_path,html_junk),"r") as file3:
                data3=file3.readlines()
            # junk_folder = os.getcwd()[:-4] + f"/database/cacheMemory"
            junk_folder = UT.getcurrentDirectory() + f"/database/cacheMemory"
            csv_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.csv')
            html_junk = os.path.join(junk_folder,f'{payload["attackName"]}{Current_Report_ID}.html')
            if os.path.exists(csv_junk):
                os.remove(csv_junk)
            if os.path.exists(html_junk):
                os.remove(html_junk)
            field_names = ['sample_index','actual_labels','final_labels','success']
            dict = np.array(payload['attack_data_status'])
            with open(csv_junk,"w",newline="") as f:
                write = csv.writer(f)
                write.writerow(field_names)
                write.writerows(dict) 
            a=pd.read_csv(csv_junk)
            a.to_html(os.path.join(report_path,html_junk))
            with open(os.path.join(report_path,html_junk),"r") as file4:
                data4=file4.readlines()
            # generating graph for particular attack  
            csv_path = os.path.join(report_path,csvfilename)
            df = pd.read_csv(csv_path)
            cols = list(df.columns)
            key = 0
            if data['groundTruthClassLabel'] in cols and 'prediction' in cols:
                graph_html = UT.graphForAttack({'folder_path':report_path, 'target':data['groundTruthClassLabel'], 'attackName':payload["attackName"]})
                key = 1   
            with open(os.path.join(report_path,f"report.html"),"w") as file:
                file.writelines(UT.htmlCssContent())
                file.writelines('<body>')
                file.writelines(f'<h2>{payload["attackName"]}_Attack</h2>')
                file.writelines(UT.htmlContent(payload["attackName"]))
                if key == 1:
                    file.writelines(f"<h3>Visualisation for Prediction Data</h3>")
                    file.writelines(graph_html)
                file.writelines("<h3>Attack Status</h3>")
                file.writelines(data3)
                if len(payload['attack_data_status']) != 0:
                    file.writelines("<h3>Adversial Input used for attack and Adversial output generated</h3>")
                    file.writelines(data4)
                file.writelines('</body>')
            shutil.make_archive(report_path,'zip',report_path)
            UT.updateCurrentID()
            UT.databaseDelete(csv_junk)
            UT.databaseDelete(html_junk)
            return foldername
        
        except Exception as exc:
            if(telemetry_flg == 'True'):
                with con.ThreadPoolExecutor() as executor:
                    executor.submit(log.log_error_to_telemetry, "generatecsvreportartendpoint", exc, apiEndPoint, errorRequestMethod)
            raise Exception
                    