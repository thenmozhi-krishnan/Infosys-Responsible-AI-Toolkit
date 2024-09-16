# Responsible-AI-Explain

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Set Configuration Variables](#set-configuration-variables)
- [Running the Application](#running-the-application)
- [License](#license)
- [Contact](#contact)

## Introduction

**Explainability** offers comprehensive explanations for traditional machine learning and deep learning models. It supports both local and global explainable AI methods, including anchors and feature importance, across classification, regression, and time series tasks. This ensures transparency and insight into model behavior, facilitating better decision-making and trust.

## Features

**Types of Explainability**

Explainability methods can be broadly categorized into **Local** and **Global** approaches.

**Local Explainability**
focuses on explaining individual predictions made by a model. It helps users understand why a particular decision was made in a specific instance. Some popular local explainability methods include:

- **LIME (Local Interpretable Model-agnostic Explanations):**

    - **How it works**: LIME approximates the complex model with a simpler, interpretable model (like a linear model) locally around the instance of interest. It perturbs the input data and observes how the model's predictions change to understand which features are important for the prediction.

- **SHAP (SHapley Additive exPlanations):**

    - **How it works**: SHAP values are derived from cooperative game theory, where the importance of each feature is determined based on its contribution to the prediction, considering all possible combinations of features.

- **Anchors:**

    - **How it works**: Anchors provide rules that “anchor” the prediction to a specific explanation. They identify feature values that are sufficient to explain the prediction with high confidence.

**Global Explainability**
aims to provide a comprehensive understanding of the overall behavior and structure of a model. It helps in understanding how the model generally makes decisions across the entire dataset. Some global explainability methods include:

- **Feature Importance:**

    - **How it works**: This method evaluates the impact of each feature on the model’s predictions. Techniques include permutation importance, partial dependence variance, and SHAP importance measures.

## Installation
To run the application, first we need to install Python and the necessary packages:

1. Install Python (version >= 3.9) from the [official website](https://www.python.org/downloads/) and ensure it is added to your system PATH.

2. Install MongoDB by following the instructions on the [official MongoDB website](https://docs.mongodb.com/manual/installation/).
    ### Importing JSON Data into MongoDB

    1. Ensure MongoDB is running on your system. You can start MongoDB using the following command:
        ```sh
        mongod
        ```

    2. Import your JSON data into MongoDB using the `mongoimport` command. Replace `<database>` and `<collection>` with your actual database name and collection name. The JSON file to be imported is `Tbl_Explanation_Methods.json` located in the `responsible-ai-explain/docs` folder:
        ```sh
        mongoimport --db <database> --collection <collection> --file responsible-ai-explain/docs/Tbl_Explanation_Methods.json --jsonArray
        ```

    3. Ensure the same database name is added to the `RAI_EXPLAIN_DB` configuration variable in your `.env` file:
        ```sh
        RAI_EXPLAIN_DB=<database>
        ```

3. Clone the repository:
    ```sh
    git clone <repository-url>
    ```

4. Navigate to the `responsible-ai-explain` directory:
    ```sh
    cd responsible-ai-explain
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

7. Go to the `requirements` directory where the `requirement.txt` file is present and install the requirements:
    ```sh
    pip install -r requirement.txt
    ```
## Set Configuration Variables

After installing all the required packages, configure the variables necessary to run the APIs.

1. Navigate to the `explain` directory:
    ```sh
    cd ..
    cd src/explain
    ```

2. Locate the `.env` file in the project directory. This file contains several configuration keys. Make sure to fill in the mandatory fields as shown below (sample values are provided for reference):

    ```sh
    RAI_EXPLAIN_DB = "${rai_explain_db}"      # [Mandatory] RAI_EXPLAIN_DB = "RAI_Explain_DB"
    DB_NAME = "${dbname}"                     # [Mandatory] DB_NAME = "Rai_Usecase_DB"
    DB_USERNAME = "${username}"               # [Optional]
    DB_PWD = "${password}"                    # [Optional]
    DB_IP = "${ipaddress}"                    # [Optional]
    DB_PORT = "${port}"                       # [Optional]
    DB_TYPE ="${dbtype}"                      # [Mandatory] DB_TYPE = "mongo"
    MONGO_PATH = "mongodb://${DB_USERNAME}:${DB_PWD}@${DB_IP}:${DB_PORT}/"    # [Mandatory] MONGO_PATH = "mongodb://localhost:27017/"
    COSMOS_PATH = "${cosmos_path}"            # [Optional]
    ```
    ```sh
    ALLOWED_ORIGINS= "${allowed_origins}"     # ALLOWED_ORIGINS ="*"         
    To allow access to all sites, use the value *. Alternatively, you can specify a list of sites that should have access.
    ```

3. Replace the placeholders with your actual values.

## Running the Application

Once we have completed all the aforementioned steps, we can start the service.

1. Navigate to the `src` directory:
    ```sh
    cd ..
    ```

2. Run `main.py` file:
    ```sh
    python main.py
     ```

3. Open the following URL in your browser:
    [http://localhost:8002/rai/v1/explainability/docs](http://localhost:8002/rai/v1/explainability/docs)

For API calls, please refer to the [API Documnet](responsible-ai-explain/docs/API_Doc.pdf)

  
## License

The source code for the project is licensed under the MIT license, which you can find in the [LICENSE.md](LICENSE.md) file.

## Contact

If you have more questions or need further insights, feel free to Connect with us @ infosysraitoolkit@infosys.com