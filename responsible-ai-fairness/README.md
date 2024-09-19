# Responsible-AI-Fairness

## Table of content
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Set Configuration Variables](#set-configuration-variables)
- [Running the Application](#running-the-application)
- [License](#license)
- [Contact](#contact)


## Introduction
Responsible-ai-fairness offers solutions for Traditional AI and LLM's fairness and bias evaluations. For traditional classification problems, the training datasets and model's predictions can be analyzed and mitigated for Group Fairness. Individual Fairness analysis is also supported to get a comprehensive analysis. For Large Language Models, given text is evaluated for bias context and highlights the affected groups and bias types using GPT-4.

## Requirements
1. Python 3.9 and above
2. VSCode
3. Download bart model from [Model](https://huggingface.co/facebook/bart-large-mnli) and place it in models folder under responsible-ai-fairness

## Features
For more details refer our [User Guide](https://github.com/Infosys-AI-Cloud-MMS/responsible-ai-fairness/blob/IP-2.0.0.0-Merge/responsible-ai-fairness/docs/Fairness_API_Doc.pdf)
 
## Installation
For more details refer our [Setup Document](https://github.com/Infosys-AI-Cloud-MMS/responsible-ai-fairness/blob/IP-2.0.0.0-Merge/responsible-ai-fairness/docs/Setup%20document.pdf)

## Set Configuration Variables
For more details refer our [Setup Document](https://github.com/Infosys-AI-Cloud-MMS/responsible-ai-fairness/blob/IP-2.0.0.0-Merge/responsible-ai-fairness/docs/Setup%20document.pdf)

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

## Contact
If you have more questions or need further insights, feel free to Connect with us @ infosysraitoolkit@infosys.com

For more details refer our [Setup Document](https://github.com/Infosys-AI-Cloud-MMS/responsible-ai-fairness/blob/IP-2.0.0.0-Merge/responsible-ai-fairness/docs/Setup%20document.pdf) and [User Guide](https://github.com/Infosys-AI-Cloud-MMS/responsible-ai-fairness/blob/IP-2.0.0.0-Merge/responsible-ai-fairness/docs/Fairness_API_Doc.pdf)

