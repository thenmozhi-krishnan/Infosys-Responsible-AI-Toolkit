## Table of Contents 
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Set Configuration Variables](#set-configuration-variables)
- [Running the Application](#running-the-application)
- [License](#license)
- [Contact](#contact)

## Introduction
**LeaderBoard** Leverage publicly available datasets to benchmark the performance of large language models (LLMs) across a variety of business-relevant tasks. Provide the accuracy results of the models on these specific datasets in a clear, tabular format for easy reference.

## Features
Providing the accuracy results of the models on different datasets in tabular format for easy reference.

## Installation
To run the application, first we need to install Python and the necessary packages:

1. Install Python (version >= 3.9) from the [official website](https://www.python.org/downloads/) and ensure it is added to your system PATH.

2. Clone the repository:
    ```sh
    git clone <repository-url>
    ```

3. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

4. Activate the virtual environment:
    - On Windows:
        ```sh
        .\venv\Scripts\activate
         ```

5. Navigate to the `responsible-ai-llm-security` directory:
    ```sh
    cd responsible-ai-llm-security
    ```

6. Navigate to `src` directory:
    ```sh
    cd src
    ```

7. Now, install the requirements:
    ```sh
    pip install -r ../requirements/requirement.txt
    ```

## Set Configuration Variables
After installing all the required packages, configure the variables necessary to run the APIs.

1. Locate the `.env` file, which contains keys like the following:

    ```sh
    DB_NAME="${dbname}"
    DB_USERNAME="${username}"
    DB_PWD="${password}"
    DB_IP="${ipaddress}"
    DB_PORT="${port}"
    MONGO_PATH="mongodb://${DB_USERNAME}:${DB_PWD}@${DB_IP}:${DB_PORT}/"
    COSMOS_PATH ="${cosmos_path}"
    DB_TYPE="${dbtype}"
    DATASETCONTAINERNAME ="${dataset_container}"
    MODELCONTAINERNAME ="${model_container}"
    ADDFILEURL ="${add_file_url}"
    DELETEFILEURL ="${delete_file_url}"
    GETFILEURL ="${get_data_url}"
    ERRORLOGAPI = "${error_log_api}"
    TELEMETRY_FLAG = "${telemetry_flag}"
    ALLOW_ORIGINS ="${allow_origins}"
    ```
    ```sh
    To allow access to all sites, use the value *. Alternatively, you can specify a list of sites that should have access.
    ```

2. Replace the placeholders with your actual values.

## Running the Application
Once we have completed all the aforementioned steps, we can start the service.

1. Run `main.py` file:
    ```sh
    python main.py
    ```

2. Open the following URL in your browser:
    [http://localhost:80/v1/infosys/llm/security/docs](http://localhost:80/v1/infosys/llm/security/docs)

## License
The source code for the project is licensed under MIT license, which you can find in the [LICENSE.md](LICENSE.md) file.

## Contact
If you have more questions or need further insights, feel free to Connect with us @ Infosysraitoolkit@infosys.com
