'''
MIT license https://opensource.org/licenses/MIT
Copyright 2024 Infosys Ltd
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import requests
import json
import base64
 
def mixtralPredict7B(prompt):
    url = "https://api-aicloud.ad.infosys.com/v1/language/generate/models/mistral-7b/versions/1/infer"
    headers = {
        'Content-Type': 'application/json'
    }
 
 
    prompt = base64.b64encode(prompt.encode("utf-8")).decode("utf-8")
   
    data = {
        "inputs": [prompt],
        "parameters": {
            "max_new_tokens": 100
        }
    }
 
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    data = response.json()
    decoded_bytes = base64.b64decode(data[0]["generated_text"])
    decoded_string = decoded_bytes.decode("utf-8")
    return decoded_string
  
print(mixtralPredict7B("classify the sentece as positive or negative"))