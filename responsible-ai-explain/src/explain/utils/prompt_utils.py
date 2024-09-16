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

class Prompt:
        
    def get_classification_prompt(input_prompt):
        
        template = f"""
            Imagine you are a Responsible AI expert with experience in explaining why a model has made a decision. Your task is to determine the sentiment of the given prompt, identify the keywords the model used to arrive at that sentiment, and provide a clear explanation of why the model classified the prompt as that sentiment.
 
            Make sure the response is simple and easy to understand. Use polite language. Do not write as a third person. Do not include Certainly! at the beginning of your response, just give response.
            
            For example:
            Prompt: Unfortunately the movie served with hacked script but the actors performed well

            Sentiment: Negative
            Keywords: ['unfortunately', 'hacked']
            Explanation: The model predicted the input as Negative because it anchored or focused on the word "unfortunately" and "hacked" in your input. In this context, the model likely learned that when someone mentions a hacked script or unfortunately, it usually indicates a negative opinion about a movie. Despite the actors performing well, the emphasis on the script being hacked led the model to classify the overall sentiment as negative. This shows that the model paid close attention to specific keywords, like "unfortunately" and "hacked" to make its prediction.
            
            similarly, provide sentiment, keywords identified to determine the sentiment and Explanation for the below given information.
            
            Input:
            Prompt: {input_prompt}
            
            Output format:
            {{
                "Sentiment": ""
                "Keywords": []
                "Explanation": ""
            }}
            """
        return template