## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Set Configuration Variables](#set-configuration-variables)
- [Running the Application](#running-the-application)
- [License](#license)
- [Contact](#contact)

## Introduction
**AI text detector** It is a tool designed to distinguish between human-written text and machine-generated text using the concept of perplexity. Perplexity is a measure of how surprised a language model is by a given sequence of words. By leveraging this measure, AI text detector calculates a score that indicates the likelihood of a text being human-written.

## Features
AI text detector different features are as follows:-
- **Baseline Perplexity** : Calculate the perplexity of a given text using a language model. This represents how surprised the model would be if it were generating the text itself.
- **Observed Perplexity** : Compute the perplexity of the actual text using the same language model.
- **Normalization** : Normalize the observed perplexity by the baseline perplexity to account for the inherent difficulty of the text.
- **AI text detector Score** : The normalized perplexity is the AI text detector score. A higher AI text detector score indicates that the text is more likely to be human-written, as it was more surprising to the language model than expected.

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

5. Navigate to the `responsible-ai-security-text-detector` directory:
    ```sh
    cd responsible-ai-security-text-detector
    ```

8. Navigate to `src` directory:
    ```sh
    cd src
    ```

9. Now, install the requirements:
    ```sh
    pip install -r ../requirements/requirement.txt
    pip install spacy
    ```

## Set Configuration Variables
After installing all the required packages, configure the variables necessary to run the APIs.

1. Locate the `.env` file, which contains keys like the following:

    ```sh
    DB_NAME="${dbname}"        # [Mandatory] DB_NAME = "raitext"
    WORKERS="${workers}"       # [Mandatory] WORKERS = "1"
    DB_USERNAME="${username}"  # [Optional]
    DB_TYPE ="${dbtype}"       # [Mandatory] DB_TYPE = "mongo"
    DB_PWD="${password}"       # [Optional]
    DB_IP="${ipaddress}"       # [Optional]
    DB_PORT="${port}"          # [Optional]
    MONGO_PATH="mongodb://${DB_USERNAME}:${DB_PWD}@${DB_IP}:${DB_PORT}/"  # [Mandatory] MONGO_PATH = "mongodb://localhost:27017/"
    ALLOW_ORIGINS ="${allow_origins}"    # ALLOW_ORIGINS ="*"
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
    [http://localhost:8000/rai/v1/models/docs](http://localhost:8000/rai/v1/models/docs)

## License
The source code for the project is licensed under MIT license, which you can find in the [LICENSE.md](LICENSE.md) file.

## Contact
If you have more questions or need further insights, feel free to Connect with us @ Infosysraitoolkit@infosys.com
