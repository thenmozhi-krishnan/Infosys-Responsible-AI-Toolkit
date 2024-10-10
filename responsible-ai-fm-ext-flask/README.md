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

2. Install MongoDB by following the instructions on the [official MongoDB website](https://docs.mongodb.com/manual/installation/).

3. Install `pip` if it is not already installed. You can download and install it by following the instructions on the [official pip website](https://pip.pypa.io/en/stable/installation/).

4. Clone the repository:
    ```sh
    git clone <repository-url>
    ```

5. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

6. Activate the virtual environment:
    - On Windows:
        ```sh
        .\venv\Scripts\activate
         ```
    - On Linux/Mac:
        ```sh
        source venv/bin/activate
        ```

7. Navigate to the `responsible-ai-llm-security` directory:
    ```sh
    cd responsible-ai-llm-security
    ```

8. Navigate to `src` directory:
    ```sh
    cd src
    ```

9. Now, install the requirements:
    ```sh
    pip install -r ../requirements/requirement.txt
    ```

## Set Configuration Variables
After installing all the required packages, configure the variables necessary to run the APIs.

1. Locate the `.env` file, which contains keys like the following:

    ```sh
    DB_NAME="${dbname}"         # [Mandatory] DB_NAME = "raillm"
    DB_USERNAME="${username}"   # [Optional]
    DB_PWD="${password}"        # [Optional]
    DB_IP="${ipaddress}"        # [Optional]
    DB_PORT="${port}"           # [Optional]
    MONGO_PATH="mongodb://${DB_USERNAME}:${DB_PWD}@${DB_IP}:${DB_PORT}/"  # [Mandatory] MONGO_PATH = "mongodb://localhost:27017/"
    COSMOS_PATH ="${cosmos_path}"   # [Optional]
    DB_TYPE="${dbtype}"             # [Mandatory] DB_TYPE = "mongo"
    DATASETCONTAINERNAME ="${dataset_container}"   # [Optional]
    MODELCONTAINERNAME ="${model_container}"       # [Optional]
    ADDFILEURL ="${add_file_url}"                  # [Optional]
    DELETEFILEURL ="${delete_file_url}"            # [Optional]
    GETFILEURL ="${get_data_url}"                  # [Optional]
    ERRORLOGAPI = "${error_log_api}"               # [Optional]
    TELEMETRY_FLAG = "${telemetry_flag}"           # [Optional] False if do not want to connect with telemetry
    ALLOW_ORIGINS ="${allow_origins}"              # ALLOW_ORIGINS ="*"
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
