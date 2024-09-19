# Responsible-AI-Moderation Layer 

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Set Configuration Variables](#set-configuration-variables)
- [Running the Application](#running-the-application)
- [License](#license)
- [Contact](#contact)
 
## Introduction
The **Moderation Layer** module provides model-based and template-based guardrails to moderate inputs by passing them through various checks like check for toxicity, profanity, prompt injection, jailbreak, privacy etc. The module also provides additional features like chain of thoughts, chain of verifications, token importance and LLM explanation.

## Features
1. **Model-based guardrail** : Here we use traditional AI models, libraries to test the input for various checks like prompt injection, jailbreak, toxicity etc.

2. **Template-based guardrail** : Here we  are making use of a Prompt Template for each evaluation check like prompt injection, jailbreak, etc. Prompt templates are a way to define reusable structures for prompting LLMs. It allows us to create a base prompt with placeholders that can be filled with different values to generate specific outputs. 

3. **Translate option** : Given that our model-based guardrails are optimized for English, we offer a translator option. In cases where prompts are in languages other than English, the translator converts them to English before passing them to the guardrails, thus providing extra protection against Multilingual Jailbreak attacks as well.

4. **Multimodal** : We are using multimodal functionality of GPT4o combined with our model based guardrail and template based guardrail to perform moderation checks like prompt injection, jailbreak etc. for text and image uploaded, respectively.

5. **Response comparision** : We are also providing response comparision between just GPT output and output from GPT with our guardrails. 

## Installation
To run the application, first we need to install Python and the necessary packages:

1. Install Python (version >= 3.9) from the [official website](https://www.python.org/downloads/) and ensure it is added to your system PATH.

2. Clone the repository : responsible-ai-fm-ext-flask:
    ```sh
    git clone <repository-url>
    ```

3. Navigate to the `responsible-ai-fm-ext-flask` directory:
    ```sh
    cd responsible-ai-fm-ext-flask
    ```

4. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

5. Activate the virtual environment:
   On Windows:
        ```sh
        .\venv\Scripts\activate
         ```

6. Go to the `requirements` directory where the `requirement.txt` file is present and install the requirements:
    ```sh
    pip install -r requirement.txt
    ```

## Set Configuration Variables
After installing all the required packages, configure the variables necessary to run the APIs.

1. Navigate to the `src` directory:
    ```sh
    cd ..
    ```

2. Locate the `.env` file, which contains keys like the following:

  ```sh
  LOGCHECK="${log}"
  PORT='${ports}'
  VAULTNAME="${vault}"
  ISVAULT="${isvault}"
  AZURE_VAULT_TENANT_ID="${azurevaulttenantid}"
  AZURE_VAULT_CLIENT_ID="${azurevaultclientid}"
  VAULT_SECRET="${VAULT_SECRET}"
  KEYVAULTURL="${KEYVAULTURL}"
  APP_VAULT_BACKEND="${APP_VAULT_BACKEND}"
  APP_VAULT_ROLE_ID="${APP_VAULT_ROLE_ID}"
  APP_VAULT_SECRET_ID="${APP_VAULT_SECRET_ID}"
  APP_VAULT_PATH="${APP_VAULT_PATH}"
  APP_VAULT_URL="${APP_VAULT_URL}"
  APP_VAULT_KEY_MONGOUSER="${APP_VAULT_KEY_MONGOUSER}"
  APP_VAULT_KEY_MONGOPASS="${APP_VAULT_KEY_MONGOPASS}"
  APP_MONGO_HOST="${APP_MONGO_HOST}"
  APP_MONGO_DBNAME="${APP_MONGO_DBNAME}"

  TEL_FLAG="${tel_flag}"
  APP_MONGO_HOST="${APP_MONGO_HOST}"
  APP_MONGO_DBNAME="${APP_MONGO_DBNAME}"

  IS_TELEMETRY_ENDPOINT="${isendpoint}"
  IS_PDATA_ID="${ispdataid}"
  IS_PDATA_VER="${ispdataver}"
  IS_PDATA_PID="${ispdatapid}"
  IS_ENV="${isenv}"
  ETA_TELEMETRY_ENDPOINT="${etaendpoint}"
  ETA_TELEMETRY_USERNAME="${etausername}"
  ETA_TELEMETRY_PASSWORD="${etapassword}"
  TELEMETRY_ENVIRONMENT="${telemetryenviron}"
  BLOOM_ENDPOINT="${bloomendpoint}"
  LLAMA_ENDPOINT="${llamaendpoint}"
  MODEL_NAME = "${modelname}"
  OPENAI_API_TYPE = "${apitype}"

  OPENAI_API_BASE_GPT3 = "${apibase_gpt3}"
  OPENAI_API_KEY_GPT3 = "${apikey_gpt3}"
  OPENAI_API_VERSION_GPT3 = "${apiversion_gpt3}"                
  OPENAI_MODEL_GPT3 = "${openaimodel_gpt3}"

  OPENAI_API_BASE_GPT4 = "${apibase_gpt4}"
  OPENAI_API_KEY_GPT4 = "${apikey_gpt4}"
  OPENAI_API_VERSION_GPT4 = "${apiversion_gpt4}"  
  OPENAI_MODEL_GPT4 = "${openaimodel_gpt4}"

  OPENAI_API_BASE_GPT4_O = "${api_base}"
  OPENAI_API_KEY_GPT4_O = "${api_key}"
  OPENAI_API_VERSION_GPT4_O = "${api_version}"  
  OPENAI_MODEL_GPT4_O = "${model}"

  SIMILARITYMODEL="${similaritymodel}"
  RESTRICTEDMODEL="${restrictedmodel}"
  DETOXIFYMODEL="${detoxifymodel}"
  PROMPTINJECTIONMODEL="${promptinjectionmodel}"
  JAILBREAKMODEL="${jailbreakmodel}"
  TELEMETRY_PATH="${telemetrypath}"
  COUPLEDTELEMETRYPATH="${coupledtelemetrypath}"
  ADMINTEMPLATEPATH="${adminTemplatepath}"
  DB_USERNAME="${username}"
  DB_PWD="${password}"
  DBTYPE="${dbtype}"
  AZURE_TRANSLATE_KEY = "${azuretranslatekey}"
  AZURE_TRANSLATE_ENDPOINT = "${azuretranslateendpoint}"
  AZURE_TRANSLATE_REGION = "${azuretranslateregion}"
  TENANT_ID = "${tenant_id}"
  CLIENT_ID = "${client_id}"
  CLIENT_SECRET = "${client_secret}"
  AUTH_URL = "${auth_url}"
  SCOPE = "${scope}"
  MONGO_PATH="mongodb://${DB_USERNAME}:${DB_PWD}@${APP_MONGO_HOST}/"
  ```

3. Replace the placeholders with your actual values.

## Running the Application
Once we have completed all the aforementioned steps, we can start the service.

1. Navigate to the `src` directory:

2. Run `main.py` file:
    ```sh
    python main.py
     ```

3. Open the following URL in your browser:
   [http://localhost:5000/rai/v1/moderations/docs](http://localhost:5000/rai/v1/moderations/docs)

For API calls, please refer to the [API Document Moderation Layer](https://infosys.atlassian.net/wiki/spaces/IF/pages/207815356/Responsible+AI+-+Tools)
  
## License
The source code for the project is licensed under the MIT license, which you can find in the [LICENSE.txt](LICENSE.txt) file.

## Contact
If you have more questions or need further insights please feel free to connect with us @ ResponsibleAI@infosys.com








