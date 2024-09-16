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

from llm_explain.utility.query_serper import GoogleSerperAPIWrapper
from llm_explain.utility import got as GraphOfThoughts
from llm_explain.utility.prompts.base import Prompt
from llm_explain.config.logger import CustomLogger
from llm_explain.utility.utility import Utils
from llm_explain.utility.azure import Azure
from sklearn.metrics.pairwise import cosine_similarity
from json.decoder import JSONDecodeError
from scipy.stats import gaussian_kde
from itertools import combinations
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from openai import AzureOpenAI
import pandas as pd
import numpy as np
import matplotlib
import asyncio
import base64
import time
import html
import json
import ast
import os

log = CustomLogger()

class ResponsibleAIExplain:

    def llm_response_to_json(response):
        """
        Converts a substring of the given response that is in JSON format into a Python dictionary.
        
        This function searches for the first occurrence of '{' and the last occurrence of '}' to find the JSON substring.
        It then attempts to parse this substring into a Python dictionary. If the parsing is successful, the dictionary
        is returned. If the substring is not valid JSON, the function will return None.
        
        Parameters:
        - response (str): The response string that potentially contains JSON content.
        
        Returns:
        - dict: A dictionary representation of the JSON substring found within the response.
        - None: If no valid JSON substring is found or if an error occurs during parsing.
        """
        try:
            result = None # Initialize result to None in case no valid JSON is found

            # Find the start index of the first '{' character and end index of the last '}' character
            start_index = response.find('{')
            end_index = response.rfind('}')

            # Check if both '{' and '}' are found and '{' comes before '}'
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_content = response[start_index:end_index+1] # Extract the substring that is potentially in JSON format
                result = json.loads(json_content) # Attempt to parse the JSON substring into a Python dictionary
            
            return result
        
        except Exception as e:
            # Log the exception if any error occurs during parsing
            log.error(f"Error parsing JSON from response: {e}", exc_info=True)
            raise

    async def analyze_heatmap(df_input):
        base64_encoded_imgs=[]
        try:
            
            df = df_input.copy()

            if "token" not in df.columns or "importance_value" not in df.columns:
                raise ValueError(
                    "The DataFrame must contain 'token' and 'importance_value' columns."
                )

            df["Position"] = range(len(df))
            
            # Calculate histogram data
            hist, bin_edges = np.histogram(df["importance_value"], bins=20)
            # Get the viridis colormap
            viridis = plt.get_cmap("viridis")
            # Initialize the figure
            fig = go.Figure()
            
            # Create the histogram bars with viridis coloring
            for i, freq in enumerate(hist):
                color = f"rgb({int(viridis(i / (len(bin_edges) - 1))[0] * 255)}, {int(viridis(i / (len(bin_edges) - 1))[1] * 255)}, {int(viridis(i / (len(bin_edges) - 1))[2] * 255)})"
                fig.add_trace(
                    go.Bar(
                        x=[(bin_edges[i] + bin_edges[i + 1]) / 2],
                        y=[freq],
                        width=np.diff(bin_edges)[i],
                        marker=dict(color=color),
                    )
                )
            
            # Calculate and add the KDE line
            x_kde = np.linspace(min(df["importance_value"]), max(df["importance_value"]), 500)
            kde = gaussian_kde(df["importance_value"])
            y_kde = kde(x_kde) * sum(hist) * (bin_edges[1] - bin_edges[0])
            fig.add_trace(
                go.Scatter(
                    x=x_kde, y=y_kde, mode="lines", line_shape="spline", line=dict(color="red")
                )
            )
            # Additional styling
            fig.update_layout(
                title=" Distribution of Importance Scores",
                title_font={'size': 25},
                xaxis_title="Importance Value",
                yaxis_title="Frequency",
                showlegend=False,
            )
            
            img_bytes = fig.to_image(format="png")
            
            
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
           
            base64_encoded_imgs.append(img_base64)
            
            # Normalize the importance values
            min_val = df["importance_value"].min()
           
            max_val = df["importance_value"].max()
            
            normalized_values = (df["importance_value"] - min_val) / (max_val - min_val)
            
            # Initialize the figure
            fig = go.Figure()
            
            # Create the bars, colored based on normalized importance_value
            for i, (token, norm_value) in enumerate(zip(df["token"], normalized_values)):
                color = f"rgb({int(viridis(norm_value)[0] * 255)}, {int(viridis(norm_value)[1] * 255)}, {int(viridis(norm_value)[2] * 255)})"
                fig.add_trace(
                    go.Bar(
                        x=[i],  # Use index for x-axis
                        y=[df["importance_value"].iloc[i]],
                        width=0.9,  # Set the width to make bars touch each other
                        marker=dict(color=color),
                    )
                )
            # Additional styling
            fig.update_layout(
                title="Importance Score per Token",
                title_font={'size': 25},
                xaxis_title="Token",
                yaxis_title="Importance Value",
                showlegend=False,
                bargap=0,  # Remove gap between bars
                xaxis=dict(  # Set tick labels to tokens
                    tickmode="array",
                    tickvals=list(range(len(df["token"]))),
                    ticktext=list(df["token"]),
                ),
                autosize=False,  # Disable automatic sizing
                width= max(10, len(df["token"]) * 0.3) * 100,  # Convert to pixels
            )
            # Rotate x-axis labels by 45 degrees
            fig.update_xaxes(tickangle=-45)
            
            img_bytes = fig.to_image(format="png")
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            base64_encoded_imgs.append(img_base64)

            top_10_important = df.nlargest(10, 'importance_value')
            top_10=top_10_important.to_dict(orient='records')

            # Extract the importance scores
            importance_values = df["importance_value"].values

            # Normalize the importance scores to be between 0 and 1
            min_val = np.min(importance_values)
            max_val = np.max(importance_values)

            if max_val - min_val != 0:
                normalized_importance_values = (importance_values - min_val) / (max_val - min_val)
            else:
                normalized_importance_values = np.zeros_like(importance_values)

            # Generate a colormap for the heatmap
            cmap = matplotlib.colormaps["inferno"]

            # Helper function to determine the text color based on the background color
            def get_text_color(bg_color):
                brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                if brightness < 0.5:
                    return "white"
                else:
                    return "black"

            # Initialize HTML string
            html_string = ""

            # Loop over tokens and construct the HTML string
            for idx, (token, importance) in df_input.iterrows():
                rgba = cmap(normalized_importance_values[idx])
                bg_color = rgba[:3]
                text_color = get_text_color(bg_color)
                
                # Explicitly handle special characters
                token_escaped = html.escape(token).replace('`', '&#96;').replace('$', '&#36;')  # Handle backticks and dollar signs
                html_string += f"<span style='background-color: rgba({int(bg_color[0]*255)}, {int(bg_color[1]*255)}, {int(bg_color[2]*255)}, 1); color: {text_color};'>{token_escaped}</span> "
            
            return top_10,base64_encoded_imgs,html_string
        except Exception as e:
            log.error(e, exc_info=True)
            raise


    async def calculate_uncertainty(n : int, prompt: str):
        try:
            max_tokens=1000
            client = AzureOpenAI(
                    api_key = os.getenv("AZURE_OPENAI_API_KEY") ,
                    api_version = os.getenv("AZURE_OPENAI_API_VERSION") ,
                    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") 
                )

            try:
                response = client.chat.completions.create(
                    n=n,
                    model=os.getenv("AZURE_DEPLOYMENT_ENGINE"), # model = "deployment_name".
                    messages=[
                        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                        {"role": "user", "content": prompt}
                    ],
                    logprobs=True,
                    top_logprobs=2,
                    max_tokens=100 
                )
            except Exception as e:
                log.error("Error occurred while calling the AzureOpenAI API", exc_info=True)
                raise Exception                                                                     
            cc=response.choices
            response_object ={}
            choices = []
            for i,c in enumerate(cc):
                contents=c.logprobs.content
                choice_i={
                    "text": c.message.content
                }
                logprobs = {}
                token_logprobs = []
                tokens=[]
                top_logprobs=[]
                for content in contents:
                    token_logprobs.append(content.logprob)
                    temp={}
                    tokens.append(content.token)
                    top_props=content.top_logprobs
                    for k in top_props:
                        temp[k.token]=k.logprob
                    top_logprobs.append(temp)
                logprobs["token_logprobs"]=token_logprobs
                logprobs["tokens"]=tokens
                logprobs["top_logprobs"]=top_logprobs
                choice_i["logprobs"]=logprobs
                choice_i["index"]=i
        
                choices.append(choice_i)
            response_object["choices"]=choices
    
            entropies = []
            distances = []
            choice_embeddings = []
            choice_embedding_tasks = [Utils.get_embedding(choice['text']) for choice in response_object['choices']]
            choice_embeddings = await asyncio.gather(*choice_embedding_tasks)

            async def process_choice(choice, choice_embedding):
                top_logprobs_list = choice['logprobs']['top_logprobs']
                mean_cosine_distances = []
                normalized_entropies = []
            
                tasks = [Utils.process_token_async(i, top_logprobs_list, choice, choice_embedding, max_tokens) for i in range(len(top_logprobs_list))]
                results = await asyncio.gather(*tasks)
            
                for mean_distance, normalized_entropy in results:
                    mean_cosine_distances.append(mean_distance)
                    normalized_entropies.append(normalized_entropy)
            
                return mean_cosine_distances, normalized_entropies
        
    
            choice_tasks = [process_choice(choice, emb) for choice, emb in zip(response_object['choices'], choice_embeddings)]
            results = await asyncio.gather(*choice_tasks)
        
    
            for mean_cosine_distances, normalized_entropies in results:
                distances.append(mean_cosine_distances)
                entropies.append(normalized_entropies)
        
            choice_distances = []
            for emb1, emb2 in combinations(choice_embeddings, 2):
                cosine_sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
                choice_distances.append(1 - cosine_sim)
            mean_choice_distance = np.mean(choice_distances)
            uncertainty_scores = {'entropies': entropies, 'distances': distances, 'mean_choice_distance': mean_choice_distance}
            return Utils.display_metrics(uncertainty_scores, response_object, n)
        except Exception as e:
            log.error(e, exc_info=True)
            raise Exception
        
    @staticmethod
    def normalize_scores(dict_list):
        # Calculate the total sum of all importance scores
        total_sum = sum(d['importance_score'] for d in dict_list)
        
        # If the total sum is zero, return the original list (to handle cases where all scores are zero)
        if total_sum == 0:
            return dict_list

        # Normalize the scores to ensure their sum equals 100
        normalized_scores = [round((d['importance_score'] / total_sum) * 100) for d in dict_list]
        
        # Adjust the scores to ensure their sum is exactly 100
        adjustment = 100 - sum(normalized_scores)
        normalized_scores[0] += adjustment

        # Update the original list with normalized scores
        for i, d in enumerate(dict_list):
            d['importance_score'] = normalized_scores[i]
        
        return dict_list
        
    def sentiment_analysis(text: str, class_names):
        log.info("Running local_explain")
        try:
            explanation = Azure().generate(Prompt.get_classification_prompt(text))
            explanation = ResponsibleAIExplain.llm_response_to_json(explanation)

            # Normalize the importance scores to ensure their sum equals 100
            explanation['token_importance_mapping'] = ResponsibleAIExplain.normalize_scores(explanation['token_importance_mapping'])

            return {"predictedTarget": explanation['Sentiment'], 
                    "anchor": explanation['Keywords'], 
                    "explanation": explanation['Explanation'],
                    "token_importance_mapping": explanation['token_importance_mapping']}
        except Exception as e:
            log.error(e,exc_info=True)
            raise

    async def local_explanation(prompt: str, response: str):
        try:
            explanation = Azure().generate(Prompt.get_local_explanation_prompt(prompt, response))
            explanation = ResponsibleAIExplain.llm_response_to_json(explanation)

            return explanation
        except Exception as e:
            log.error(e, exc_info=True)
            raise
        
    async def process_importance(importance_function, *args, **kwargs):
        try:
            importance_map = await importance_function(*args, **kwargs)
            importance_map_df = pd.DataFrame(importance_map, columns=['token', 'importance_value'])
            offset = importance_map_df['importance_value'].mean()

            importance_log = Utils.scale_importance_log(
                importance_map, 
                base=None, 
                offset=offset, 
                min_percentile=0,
                max_percentile=100, 
                scaling_factor=1, 
                bias=0
            )
            importance_log_df = pd.DataFrame(importance_log, columns=['token', 'importance_value'])
            
            return importance_log_df
        
        except Exception as e:
            log.error(e, exc_info=True)
            raise 
    
    async def prompt_based_token_importance(prompt):
       
        try:
            max_retries = 5
            attempts = 0
            while attempts < max_retries:
                try:
                    explanation = Azure().generate(Prompt.get_token_importance_prompt(prompt))
                    # Manually find the JSON substring within the mixed content
                    start_index = explanation.find('{')
                    end_index = explanation.rfind('}')
                    if start_index != -1 and end_index != -1 and end_index > start_index:
                        json_content = explanation[start_index:end_index+1]
                        result = json.loads(json_content)
                        # If JSON loads successfully, break out of the loop
                        break
                except JSONDecodeError:
                    attempts += 1
                    if attempts == max_retries:
                        raise Exception("Failed to decode JSON after 5 attempts.")
                    else:
                        log.debug(f"JSONDecodeError encountered. Retrying... Attempt {attempts}/{max_retries}")
                        # Add a delay before the next attempt
                        time.sleep(2) # Delay for 2 seconds
                        
            # Assuming 'result' is a dictionary with "Token" and "Importance Score" as keys, and their values are lists
            # First, create a DataFrame from the 'result' dictionary
            tokens = result['Token']
            scores = result['Importance Score']
            positions = list(range(1, len(result['Token']) + 1))

            # Find the length of the shortest list
            min_length = min(len(tokens), len(scores), len(positions))

            # Trim the lists to the length of the shortest list
            tokens = tokens[:min_length]
            scores = scores[:min_length]
            positions = positions[:min_length]

            df = pd.DataFrame({
                'token': tokens,
                'importance_value': scores,
                'position': positions
            })
 
            df['importance_value'] = df['importance_value'].astype(float)
 
            # Sort the DataFrame by 'Importance Score' in descending order to get the most important tokens first
            df_sorted = df.sort_values(by='importance_value', ascending=False)
 
            # Select the top 10 important tokens
            df_top10 = df_sorted.head(10)
            df_top10.reset_index(drop=True, inplace=True)
           
            top_10, base64_encoded_imgs, token_heatmap = await ResponsibleAIExplain.analyze_heatmap(df_top10[['token', 'importance_value']])
 
            return df_top10.to_dict(orient='records'), base64_encoded_imgs, token_heatmap
        
        except Exception as e:
            log.error(e, exc_info=True)
            raise 

    async def graph_of_thoughts(prompt: str, modelName: str):
        try:
            budget = 30
            task = "answer the following question"
            question = prompt
            approaches = [GraphOfThoughts.got]
            modelName = modelName
            
            formatted_graph, formatted_thoughts = GraphOfThoughts.run(task=task, question=question, 
                                                                      methods=approaches, 
                                                                      budget=budget, 
                                                                      lm_name=modelName)
            
            formatted_graph[3]['operation'] = 'final_thought'
            for i in range(4):
                thoughts = formatted_graph[i]['thoughts']
                for j in range(len(thoughts)):
                    formatted_graph[i]['thoughts'][j]['score'] = round(formatted_graph[i]['thoughts'][j]['score'], 2)

            return formatted_graph, formatted_thoughts
        except Exception as e:
            log.error(e, exc_info=True)
            raise

    async def search_augmentation(inputPrompt, llmResponse):
        try:
            import datetime
            current_date = datetime.datetime.now()

            # Step 1: Generate Facts with LLM response
            facts = Azure().generate(Prompt.generate_facts_prompt(inputPrompt, llmResponse, current_date))
            if isinstance(facts, str):
                facts = ResponsibleAIExplain.llm_response_to_json(facts)
            facts_list = [fact['Fact'] for fact in facts['Facts']] # Extracting the facts into a list of strings

            # Step 2: Filter the facts that are relevant to the input prompt
            filtered_facts = Azure().generate(Prompt.filter_facts_prompt(prompt=inputPrompt, facts=facts_list))
            filtered_facts = ast.literal_eval(filtered_facts)
            filtered_facts_ir = [fact + ' is this statement valid as of today ? why ? #' for fact in filtered_facts]
            questions = [inputPrompt] + filtered_facts_ir

            # Step 3: Run the prompt and facts through the Google Serper API
            search = GoogleSerperAPIWrapper()
            internet_responses = await search.run([inputPrompt])
            answers = [item[0]['content'] for item in internet_responses]
            qa_dict_list_prompt = [{'question': q, 'answer': a} for q, a in zip([inputPrompt], answers)] # Creating the list of dictionaries

            internet_responses = await search.run(questions)
            answers_facts = [item[0]['content'] for item in internet_responses]
            qa_dict_list = [{'question': q, 'answer': a} for q, a in zip(questions, answers_facts)] # Creating the list of dictionaries

            if len(facts_list) == 0:
                return {'internetResponse': qa_dict_list_prompt[0]['answer'], 
                        'factual_check': {"Score": 0.0,
                                          "explanation_factual_accuracy": {'Result': ['No facts found in the LLM response.']} }}

            # Step 4: Summarize the internet responses for prompt and facts
            summary_prompt = Azure().generate(Prompt.summarize_prompt(qa_dict_list_prompt))

            # Step 5: Evaluate fact with Google Search results
            facts = Azure().generate(Prompt.evaluate_facts_prompt(facts=filtered_facts_ir, context=qa_dict_list, prompt=inputPrompt))
            if isinstance(facts, str):
                facts = ResponsibleAIExplain.llm_response_to_json(facts)

            # In facts['Result'], each fact is a dictionary with keys 'Fact', 'Reasoning', and 'Judgement', update Fact with the filtered facts
            for i, fact in enumerate(facts['Result']):
                fact['Fact'] = filtered_facts[i]

            factuality_check = { "Score": 1.0,
                                "explanation_factual_accuracy": facts }

            return {'internetResponse': summary_prompt,
                    'factual_check': factuality_check}
        
        except Exception as e:
            log.error(e, exc_info=True)
            raise 