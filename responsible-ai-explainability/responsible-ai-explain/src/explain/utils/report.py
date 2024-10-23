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
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import base64
from datetime import datetime
from explain.constants.local_constants import ANCHOR_EXPLANATION, INTEGRATED_GRADIENT_EXPLANATION, FEATURE_IMPORTANCE_EXPLANATION, \
    FEATURE_IMPORTANCE_GLOBAL_EXPLANATION

class Report:
    
    global css_content
    
    def htmlCssContent():

        html_data = """
            <style>
                h2{
                    text-align: right;
                    background-color: #8626C3;
                    color: black;
                    font-size: 35px;
                    font-family: sans-serif;
                }
                h3{
                    color: darkorchid;
                    text-align: left;
                    font-family: sans-serif;
                }
                table{
                    border-collapse: collapse;
                    width: 50%;
                    text-align: center;
                    font-family: sans-serif;
                }
                table,th,td{
                    border: 2px solid #000000;
                }
                th{
                    background-color: #f2f2f2;
                    text-align: center;
                    font-family: sans-serif;
                }
                .header{
                    color: darkorchid;
                    font-size: 25px;
                    margin: inherit;
                    font-style: normal;
                    margin-top: 25px;
                    font-family: sans-serif;
                }
                .note{
                    background-color: #f9f9f9;
                    padding: 10px;
                    border-left: 5px solid #ccc;
                    margin-bottom: 20px;
                    line-height: 1.6;
                    font-family: sans-serif;
                }
                .content{
                    font-weight: normal;
                    color: black;
                    font-size: 17px; 
                    font-family: sans-serif;
                }
                .main-content{
                    font-weight: normal;
                    color: black;
                    font-size: 17px; 
                    font-family: sans-serif;
                }
                .scope{
                    color: darkorchid;
                    text-align: left;
                    font-size: 25px; 
                    font-family: sans-serif;
                }
                .data-container{
                    width: 80%;
                    margin: 30px;
                }
                .data-label, .data-value{
                    display: inline-block;
                    width: 25%;
                    font-family: sans-serif
                    text-align: left;
                }
                .data-label{
                    margin-right: 15px;
                }
                .container{
                    padding: 20px;
                    margin-top: -20px;
                    font-size: 20px;
                    font-family: sans-serif;
                }
                .face-card{
                    display: inline-block;
                    border: 1px solid #ccc;
                    padding: 20px;
                    width: calc(33.33% -20px);
                    text-align: center;
                    font-family: sans-serif;
                    box-shadow: 0 8px 12px rgb(5,30,217,0.94);
                    margin-top: 10px;
                    margin-left: 45px;
                    margin-right: 5px;
                    width: 20%;
                    background-color: #e4eae5;
                }
                .success{
                    color: green;
                }
                .fail{
                    color: red;
                }
                .pending{
                    color: orange;
                }
                .date-time {
                    text-align: right;
                    
                }
            </style>
        """

        return html_data
    
    css_content = htmlCssContent()
    
    def how_to_read(method_name: str):
        
        if method_name == 'ANCHOR':
            explanation = ANCHOR_EXPLANATION
        elif method_name == 'INTEGRATED GRADIENTS':
            explanation = INTEGRATED_GRADIENT_EXPLANATION
        elif method_name == 'SHAP' or method_name == 'PERMUTATION IMPORTANCE' or method_name == 'PARTIAL DEPENDENCE VARIANCE':
            explanation = FEATURE_IMPORTANCE_EXPLANATION
        elif method_name == 'Global Tree SHAP':
            explanation = FEATURE_IMPORTANCE_GLOBAL_EXPLANATION
        
        html_content = f"""<style>{css_content}<body>
                                                <div>
                                                    <p class="main-content">{explanation}</p>
                                                </div>
                                            </body>
                                        """
        
        return html_content
    
    def model_prediction(item: dict):
        
        html_content = f"""<style>{css_content}</style>
            <table border="1" class="modelDetails" >
            <thead>
                <tr style="text-align: right; color: darkorchid">
                    <th>Model Prediction</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>{item.modelPrediction}</td>
                </tr>
            </tbody>
            </table>
            """
        return html_content

    def tabular_input(input_rows: list, target_label: str = None, model_predictions: list = None, start_idx: int = 0):
        html_content = f"""<style>{css_content}</style>
                            <p style="font-family: sans-serif; margin-top: 20px; color: darkorchid"><strong>Actual Data:</strong></p>
                            <table border="1" class="inputRow" style="width: 85%;">
                            <thead>
                                <tr style="text-align: center;">
                                    <th>Row Number</th>
                                    <th>{target_label + " (target)"}</th>
                                    {''.join([f'<th>{row["featureName"]}</th>' for row in input_rows[0]])}
                                </tr>
                            </thead>
                            <tbody>
                        """
        # Accumulate rows inside the loop
        for i, input_row in enumerate(input_rows):
            start_idx += 1
            model_prediction = model_predictions[i] if model_predictions else ""
            html_content += f"""<tr style="text-align: center;">
                                    <td>{start_idx}</td>
                                    <td>{model_prediction}</td>
                                    {''.join([f'<td>{round(row["featureValue"], 2)}</td>' for row in input_row])}
                                </tr>
                            """
        
        # Close the table structure
        html_content += """</tbody>
                            </table></br>
                        """
        return html_content

    def text_input(input_text: str):
        
        html_content = f"<body><style>{css_content}</style><h4 class='main-content'>{input_text}</h4></body>"
        
        return html_content

    def create_graph_with_summary(data, path, method):

        # Set the style for the plot
        sns.set_style(style="white")

        data_list = data
        # Determine the number of subplots needed
        num_plots = len(data)
        num_cols = (num_plots + 2) // 3  
        
        # Create a figure with subplots
        fig, axes = plt.subplots(3, num_cols, figsize=(7 * num_cols, len(data[0].feature) * 0.5 * 3))
        
        # Flatten the axes array for easy iteration
        axes = axes.flatten()
        
        for i, data in enumerate(data):
            ax = axes[i]
            
            # Define a custom color palette for LIME TABULAR
            def custom_palette(values):
                return ['green' if v >= 0 else 'red' for v in values]
            
            # Define the threshold
            threshold = 20

            # Filter features based on the threshold
            filtered_data = data[abs(data['value']) >= threshold]

            # Exclude the "others" feature and extract the feature names
            feature_names = filtered_data[filtered_data['feature'] != 'Others']['feature'].tolist()

            # Exclude the "Others" feature and separate positive and negative contributions
            positive_contributions = ', '.join(filtered_data[(filtered_data['value'] > 0) & (filtered_data['feature'] != 'Others')]['feature'].tolist()).replace("'", '')
            negative_contributions = ', '.join(filtered_data[(filtered_data['value'] < 0) & (filtered_data['feature'] != 'Others')]['feature'].tolist()).replace("'", '')
            
            # Check if the method is "LIME TABULAR" and apply the custom palette
            if method == "LIME TABULAR":
                palette = custom_palette(data.value)
                sns.barplot(x=data.value, y=data.feature, data=data, palette=palette, orient="h", ax=ax)
                ax.set_xlim(data.value.min() - 20, data.value.max() + 20)
                ax.set_ylabel('Feature Thresholds')
                ax.set_title(f'Feature Importance for record - {i + 1}')
            else:
                sns.barplot(x=data.value, y=data.feature, data=data, color='#0072BD', orient="h", ax=ax)
                ax.set_xlim(0, 100)
                ax.set_ylabel('Feature Names')
                ax.set_title(f'Feature Importance')
            
            ax.set_xlabel('Contribution (%)')
            
            # Get current y ticks
            yticks = ax.get_yticks()

            # Set y ticks
            ax.set_yticks(yticks)

            # Rotate category labels
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

            # Add the value on top of each bar
            for p in ax.patches:
                width = p.get_width()
            
                # Only add text if the width is greater than 0
                if width > 0:
                    ax.text(width, p.get_y() + p.get_height() / 2, ' {:1.2f}%'.format(width), ha='left', va='center')
                if width < 0:
                    ax.text(width, p.get_y() + p.get_height() / 2, ' {:1.2f}%'.format(width), ha='right', va='center')
            
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout to add space between the border and the graph
        plt.tight_layout()

        # Save the plot to the specified path
        plt.savefig(path, format='png', bbox_inches='tight')
        plt.close()

        # Convert the image to base64
        with open(path, "rb") as imagefile:    
            graph_url = base64.b64encode(imagefile.read()).decode('utf-8')

        html_content = f"<style>{css_content}</style><body><img src='data:image/png;base64,{graph_url}' alt={method}></body>"

        for i, data in enumerate(data_list):
                
                # Define a custom color palette for LIME TABULAR
                def custom_palette(values):
                    return ['green' if v >= 0 else 'red' for v in values]
                
                # Define the threshold
                threshold = 20

                # Filter features based on the threshold
                filtered_data = data[abs(data['value']) >= threshold]

                # Check if filtered_data is empty
                if filtered_data.empty:
                    # Take the first feature from the original data
                    first_feature = data.iloc[0]['feature']
                    filtered_data = data[data['feature'] == first_feature]
                
                # Exclude the "others" feature and extract the feature names
                feature_names = filtered_data[filtered_data['feature'] != 'Others']['feature'].tolist()

                # Exclude the "Others" feature and separate positive and negative contributions
                positive_contributions = ', '.join(filtered_data[(filtered_data['value'] > 0) & (filtered_data['feature'] != 'Others')]['feature'].tolist()).replace("'", '')
                negative_contributions = ', '.join(filtered_data[(filtered_data['value'] < 0) & (filtered_data['feature'] != 'Others')]['feature'].tolist()).replace("'", '')   

                if method != "LIME TABULAR":
                    feature_names = ', '.join(feature_names).replace("'", '')
                    verb = "is" if len(feature_names.split(', ')) == 1 else "are"
                    html_content += f"""
                                <div style="font-family: sans-serif; margin-top: 20px;">
                                    <p> From the results, <strong>{feature_names}</strong> {verb} contributing more to the model's predictions.</p>
                                </div>
                            """
                else:
                    positive_verb = "is" if len(positive_contributions.split(', ')) == 1 else "are"
                    negative_verb = "is" if len(negative_contributions.split(', ')) == 1 else "are"

                    html_content += f"""
                        <div style="font-family: sans-serif; margin-top: 20px; padding-left: 40px;">
                            <p><strong>Graph {i + 1} Analysis:</strong></p>
                    """
                    if positive_contributions:
                        html_content += f"""
                                        <p><strong style="color: green;">Positive Influence:</strong> <strong>{positive_contributions}</strong> {positive_verb} contributing positively. This means that these features increase the likelihood of the predicted outcome.</p>
                                    """
                    if negative_contributions:
                        html_content += f"""
                                        <p><strong style="color: red;">Negative Influence:</strong> <strong>{negative_contributions}</strong> {negative_verb} contributing negatively. This indicates that these features decrease the likelihood of the predicted outcome.</p>
                                    """
                    html_content += "</div>"

        return html_content

    def generate_html_content(json_obj):
        
        # Get current date and time
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y %I:%M:%S %p")

        html_content = "<style>p, div, h1, h2, h3, h4, h5, h6 { page-break-inside: avoid; }</style>"
        html_content += f"<div class='date-time'>{date_time}</div>"
        html_content += f"<h2 style='text-align:center; color:white; background-color:darkorchid; font-size:35px; font-family: sans-serif;'>Explainability Report</h2>"
        
        # Initialize a flag to False
        local_explainability_added = False
        global_explainability_added = False
        model = False
        
        for item in json_obj:
            if not model:
                html_content = html_content + f"<h3 class=scope>Model Information</h3>"
                html_content += "<table style='border: none; border-collapse: collapse; width: 75%;'>"
                html_content += "<tr>"
                html_content += f"<td style='border: none; font-family: sans-serif; text-align: left; margin-right: 20px; white-space: nowrap; padding-right: 20px;'><strong>Usecase:</strong></td>"
                html_content += f"<td style='border: none; font-family: sans-serif; text-align: left; margin-right: 20px;'>{item['title']}</td>"
                html_content += "</tr>"
                html_content += "<tr>"
                html_content += f"<td style='border: none; font-family: sans-serif; text-align: left; margin-right: 20px; white-space: nowrap; padding-right: 20px;'><strong>Model Type:</strong></td>"
                html_content += f"<td style='border: none; font-family: sans-serif; text-align: left; margin-right: 20px;'>{item['taskType']}</td>"
                html_content += "</tr>"
                if item['algorithm'] is not None:
                    html_content += "<tr>"
                    html_content += f"<td style='border: none; font-family: sans-serif; text-align: left; margin-right: 20px; white-space: nowrap; padding-right: 20px;'><strong>Algorithm:</strong></td>"
                    html_content += f"<td style='border: none; font-family: sans-serif; text-align: left; margin-right: 20px;'>{item['algorithm']}</td>"
                    html_content += "</tr>"
                if item['endpoint'] is not None:
                    html_content += "<tr>"
                    html_content += f"<td style='border: none; font-family: sans-serif; text-align: left; margin-right: 20px; white-space: nowrap; padding-right: 20px;'><strong>Endpoint:</strong></td>"
                    html_content += f"<td style='border: none; font-family: sans-serif; text-align: left; margin-right: 20px;'>{item['endpoint']}</td>"
                    html_content += "</tr>"
                html_content += "<tr>"
                html_content += f"<td style='border: none; font-family: sans-serif; text-align: left; margin-right: 20px; white-space: nowrap; padding-right: 20px;'><strong>Dataset Name:</strong></td>"
                html_content += f"<td style='border: none; font-family: sans-serif; text-align: left; margin-right: 20px;'>{item['datasetName']}</td>"
                html_content += "</tr>"
                html_content += "<tr>"
                html_content += f"<td style='border: none; font-family: sans-serif; text-align: left; margin-right: 20px; white-space: nowrap; padding-right: 20px;'><strong>Target Variable:</strong></td>"
                html_content += f"<td style='border: none; font-family: sans-serif; text-align: left; margin-right: 20px;'>{item['groundTruthLabel']}</td>"
                html_content += "</tr>"
                res = ', '.join(item['featureNames']).replace("'", '')
                html_content += "<tr>"
                html_content += f"<td style='border: none; font-family: sans-serif; text-align: left; margin-right: 20px; white-space: nowrap; padding-right: 20px;'><strong>Input Variables:</strong></td>"
                html_content += f"<td style='border: none; font-family: sans-serif; text-align: left; margin-right: 20px;'>{res}</td>"
                html_content += "</tr>"
                if item['groundTruthClassNames'] is not None:
                    res = ', '.join(item['groundTruthClassNames']).replace("'", '')
                    html_content += "<tr>"
                    html_content += f"<td style='border: none; font-family: sans-serif; text-align: left; margin-right: 20px; white-space: nowrap; padding-right: 20px;'><strong>Categories of Target:</strong></td>"
                    html_content += f"<td style='border: none; font-family: sans-serif; text-align: left; margin-right: 20px;'>{res}</td>"
                    html_content += "</tr>"
                html_content += "</table>"
                
                model = True

            if item['scope'] == 'GLOBAL' and not global_explainability_added:
                html_content += f"<style>{css_content}</style><body><div class='global-explanation'><h3 class=scope>Global Explanation: </h3></div></body>"
                html_content += f"""
                            <div style="font-family: sans-serif; margin-top: 20px;">
                                    <p>Global explanation helps to understand which features have the most significant impact on the model's predictions across the dataset.</p>
                            </div>
                        """
                # Set the flag to True so we don't add 'Global Explainability' again
                global_explainability_added = True

            if item['scope'] == 'LOCAL' and not local_explainability_added:
                html_content += f"<style>{css_content}</style><body><div class='local-explanation'><h3 class=scope>Local Explanation: </h3></div></body>"
                html_content += f"""
                            <div style="font-family: sans-serif; margin-top: 20px;">
                                    <p>Local explanation helps to understand which features had the most significant impact on the model's prediction for a specific instance.</p>
                            </div>                                
                        """
                # Set the flag to True so we don't add 'Global Explainability' again
                local_explainability_added = True
            
            if item['scope'] == 'GLOBAL':
                for key, val in item.items():
                    if key == "featureImportance" and val is not None:
                        data_list = []
                        feature, value = [], []
                        for d in item['featureImportance'][0].explanation:
                            
                            feature.append( d['featureName'])
                            value.append(d['importanceScore'])
                            data = {'feature': feature, 'value': value} 

                        data_list.append(pd.DataFrame(data))

                        html_content += f"""
                        <div style="font-family: sans-serif; margin-top: 20px;">
                            <p>The feature importance plot below shows the contribution of each feature to the model's predictions, with features on the y-axis and their contributions on the x-axis.</p>
                        </div>
                        """
                        html_content += Report.create_graph_with_summary(data=data_list, path="../output/image.png", method=item['methodName'])

            elif item['scope'] == 'LOCAL':
                for key, val in item.items():
                    if key == "featureImportance" and val is not None:
                        input_rows = []
                        model_predictions = []
                        html_content += f"""
                        <div style="font-family: sans-serif; margin-top: 20px;">
                            <p>Following section has output of local explanation for the five selected records.</p>
                            <p>The feature importance plot shows the contribution of each feature to the prediction for the instances below, with feature thresholds listed on the y-axis and their contributions on the x-axis. The contributions can be both positive and negative, indicating how each feature either increases or decreases the prediction score for these specific instances.</p>
                        </div>
                        """
                        data_list = []
                        for row_item in item['featureImportance'][0:5]:
                            feature =[]
                            value =[]
                            for d in row_item.explanation:
                                feature.append( d['featureName'])
                                value.append(d['importanceScore'])
                                data = {'feature': feature, 'value': value} 
                            data_list.append(pd.DataFrame(data))
                            # Add input row and model prediction to the lists
                            input_rows.append(row_item.inputRow)
                            model_predictions.append(row_item.modelPrediction)
                        # csv_link = f'<a href="local_explanation.csv" target="_blank">CSV Link</a>'
                        html_content += Report.create_graph_with_summary(data_list, "../output/image.png", item['methodName'])
                        html_content += Report.tabular_input(input_rows, item['groundTruthLabel'], model_predictions, start_idx=0)
                        html_content += f"""
                            <div style="font-family: sans-serif; margin-top: 20px;">
                                <p><strong>Note: </strong>Local explanation for entire input data is available in the report zip folder in CSV format.</p>
                                <p>Here is the full link: <a href="local_explanation.csv" target="_blank">local_explanation.csv</a></p>
                            </div>
                        """
                        
                    elif key == "timeSeriesForecast" and val is not None:
                        for ele in item['timeSeriesForecast']:
                            if ele.inputRow is not None:
                                html_content += Report.tabular_input(ele.inputRow)
                            html_content += Report.model_prediction(ele)
                            if item['methodName'] not in ["LIME TABULAR EXPLAINER", "SHAP EXPLAINER"]:
                                html_content += f"""<style>{css_content}<body>
                                                        <div>
                                                            <h3>How to read?</h3>
                                                            <p class="main-content">The Lime Time Series Explanation Plot provides insights into the importance of different features over time. Below are the key points to understand this graph:</p>
                                                            <h4>Subplots:</h4>
                                                            <p>&emsp;The plot consists of multiple subplots, each representing a specific features. Each subplot displays the relationship between the features and the target (forecast value).</p>
                                                            <h4>X-Axis:</h4>
                                                            <p class="main-content">&emsp;The x-axis represents the time series data points. The index values range from approximately 95 to 120, indicating specific time intervals or indices.</p>
                                                            <h4>Y-Axis:</h4>
                                                            <p class="main-content">&emsp;Each subplot has its own y-axis scale corresponding to the values of the respective features. The y-axis values vary based on the data range of each features.</p>
                                                            <h4>Lines:</h4>
                                                            <h4>&emsp;Input Time Series (Blue Line):</h4>
                                                            <p class="main-content">&emsp;Represents the actual data values of each features over time. Shows how the features changes throughout the time series.</p>
                                                            <h4>&emsp;TSLime Weights (Normalized) (Red Line):</h4>
                                                            <p class="main-content">&emsp;These bars represent the TSLime Weights (Normalized) for each data point in the time series.The normalized weights indicate the sensitivity of the model to specific features (variables) at different time points. Positive weights suggest that an increase in the feature value positively impacts the forecast, while negative weights indicate the opposite.</p>
                                                            <h4>&emsp;Relative Signed Value:</h4>
                                                            <p class="main-content">&emsp;The signed value (positive or negative) of each weight is crucial.</p>
                                                            <h4>&emsp;Positive Weight:</h4>
                                                            <p class="main-content">&emsp;If a feature has a positive weight, it means that an increase in that feature's value contributes positively to the forecast. For example, if the temperature rises, and the weight associated with temperature is positive, the model predicts a higher forecast.</p>
                                                            <h4>&emsp;Negative Weight:</h4>
                                                            <p class="main-content">&emsp;Conversely, if a feature has a negative weight, it means that a decrease in that feature's value impacts the forecast negatively. For instance, if the size of a store decreases, and the weight associated with store size is negative, the model predicts a lower forecast.</p>
                                                            <p class="main-content">In summary, the red bars in the plot represent the relative importance of different features at different time intervals. Positive weights indicate positive impact, while negative weights indicate negative impact on the forecast.</p>
                                                        </div>
                                                    </body>
                                                """
                            elif item['methodName'] == 'LIME TABULAR EXPLAINER':
                                html_content = html_content + f"""<style>{css_content}<body>
                                            <div>
                                                <h3>Understanding Lime tabular for ARIMA Model</h3>
                                                <p>This section explains the results of using Lime Tabular to understand the inner workings of your ARIMA model for time series forecasting.</p>
                                                <h3>Interpreting the Lime Tabular Plot</h3>
                                                <ul>
                                                    <li>Y-axis: Represents the features (past data points) used by the ARIMA model.</li>
                                                    <li>X-axis: Represents the importance value for each feature.</li>
                                                    <li>Bar height: Indicates the magnitude of the feature's contribution.</li>
                                                    <li>Bar color: Green indicates the feature increased the forecast, red indicates it decreased the forecast.</li>
                                                </ul>
                                            </div>
                                        </body>
                                    """
                            elif item['methodName'] == 'SHAP EXPLAINER':
                                html_content = html_content + f"""<style>{css_content}<body>
                                                <div>
                                                    <h3>Understanding Kernel SHAP for ARIMA Model</h3>
                                                    <p>This section explains the results of using Kernel SHAP to understand the inner workings of your ARIMA model for time series forecasting.</p>
                                                    <h3>Interpreting the SHAP Summary Plot</h3>
                                                    <ul>
                                                        <li>Y-axis: Represents the features (past data points) used by the ARIMA model.</li>
                                                        <li>X-axis: Represents the SHAP value for each feature.</li>
                                                        <li>Bar height: Indicates the magnitude of the feature's contribution.</li>
                                                    </ul>
                                                </div>
                                            </body>
                                        """
                            html_content += f"<style>{css_content}</style><body><img src='data:image/png;base64,{ele.timeSeries}' alt={item['methodName']}></body>"
        return html_content  