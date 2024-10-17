# Responsible-AI-Fairness

## Table of content
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Features](#features)
- [Installation & set configuration variables](#installation)
- [Running the Application](#running-the-application)
- [License](#license)
- [Open Source Tools used](#open-source-tools-used)
- [Contact](#contact)


## Introduction
Responsible-ai-fairness offers solutions for Traditional AI and LLM's fairness and bias evaluations. For traditional classification problems, the training datasets and model's predictions can be analyzed and mitigated for Group Fairness. Individual Fairness analysis is also supported to get a comprehensive analysis. For Large Language Models, given text is evaluated for bias context and highlights the affected groups and bias types using GPT-4.

## Requirements
1. Python 3.9 and above
2. pip
3. Mongo DB
4. VSCode
5. infosys_responsible_ai_fairness-1.1.5-py2.py3-none-any.whl file having code to calculate metrics scores for bias analysis using [aif360](https://aif360.readthedocs.io/en/stable/), [Holistic AI](https://github.com/holistic-ai/holisticai), [Fairlearn](https://github.com/fairlearn/fairlearn). run requirements.txt using pip install -r requirements.txt command at requirements folder path for installation.
6. aicloudlibs-0.1.0-py3-none-any.whl file having code for logging and exception handling and run requirements.txt using pip install -r requirements.txt command at requirements folder path for installation.
7. BART-large-mnli is a variant of the BART model specifically fine-tuned for multi-label natural language inference (MNLI) tasks. It features 406 million parameters, a maximum token size of 1024, 24 transformer layers, and a hidden size of 1024.  
Steps to Download BART-large-mnli:
   1.	Identify the Model URL: Navigate to the BART model page on the Hugging Face Model Hub. For example, for facebook/bart-large, the URL is:
      https://huggingface.co/facebook/bart-large
   2.	Find the Model Files: On the model page, you can see the available model files:
      
       •	pytorch_model.bin (the model weights)
      	
       •	config.json (model configuration)
 
       •	tokenizer.json or other tokenizer file.
   
   3.	Download the Files: You can use curl or wget to download the files directly from the command line.
     	
       curl -L -o pytorch_model.bin https://huggingface.co/facebook/bart-large/resolve/main/pytorch_model.bin
 
       curl -L -o config.json https://huggingface.co/facebook/bart-large/resolve/main/config.json
 
       curl -L -o tokenizer.json https://huggingface.co/facebook/bart-large/resolve/main/tokenizer.json
   
   4. place model in the models folder under responsible-ai-fairness(responsible-ai-fairness/responsible-ai-fairness/models).


## Features
For more details refer our [User Guide](responsible-ai-fairness/docs/Fairness_API_Doc.pdf)

| Model Type                      | Phase         | Function  | Description                                                                 |
|---------------------------------|---------------|-----------|-----------------------------------------------------------------------------|
| Traditional Binary classification | Pretrain      | Analyze   | Analyze for bias in structured dataset based on ground truth                |
| Traditional Binary classification | Posttrain     | Analyze   | Analyze for bias in structured dataset based on model's predictions         |
| Traditional Binary classification | In-processing | Mitigation | Create a fairness aware classification model based on sensitive attributes in the pretrain dataset |
| Traditional Binary classification | Pretrain      | Mitigation | Mitigate the bias in the pretrain dataset                                   |
| Traditional Binary classification | Individual Metric | Analyze   | Analyze for bias in structured dataset based on individuals in the dataset   |
| Large Language Model            | NA            | Analyze   | Analyze bias in given unstructured text using Open AI GPT model             |
 
## Installation
For more details refer our [Setup Document](responsible-ai-fairness/docs/Setup%20document.pdf)

## Running the Application
1. Clone this repository in vscode
2. Create a virtual environment for python using cmd -
   `python -m venv <env-name>`
3. Activate the python virtual environment and install all the dependencies in requirement.txt file of the cloned repository -
   `pip install -r responsible-ai-fairness/requirements/requirements.txt`
4. Open .env file in vscode and configure the entries in it
5. In the virtual environment go to src folder of cloned repository and run below command to run the module-
   `py main_api.py`

## License
The source code for the project is licensed under the MIT license, which you can find in the [LICENSE.md](LICENSE.md) file.

## Open Source Tools used
| Open Source Tools Used | Link |
|------------------------|------|
| IBM AiF360             | https://github.com/Trusted-AI/AIF360 |
| Holistic AI            | https://github.com/holistic-ai/holisticai |
| Microsoft Fairlearn    | https://github.com/fairlearn/fairlearn |
| Facebook BART model    | https://huggingface.co/facebook/bart-large-mnli |


## Contact
If you have more questions or need further insights, feel free to Connect with us @ infosysraitoolkit@infosys.com

For more details refer our [Setup Document](responsible-ai-fairness/docs/Setup%20document.pdf) and [User Guide](responsible-ai-fairness/docs/Fairness_API_Doc.pdf)

