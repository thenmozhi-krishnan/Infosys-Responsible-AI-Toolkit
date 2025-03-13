# Responsible-AI-Backend

## Table of content
- [Introduction](#introduction)
- [Installation](#installation)
- [Set Configuration Variables](#set-configuration-variables)
- [Running the Application](#running-the-application)
- [License](#license)
- [Contact](#contact)


## Introduction
The backend repository is used for managing Log in, Register and Authentication of the users.

## Installation
To run the application, first we need to install Python and the necessary packages:

1. Install Python (version >= 3.9) from the [official website](https://www.python.org/downloads/) and ensure it is added to your system PATH.

2. Install MongoDB by following the instructions on the [official MongoDB website](https://docs.mongodb.com/manual/installation/).

3. Install `pip` if it is not already installed. You can download and install it by following the instructions on the [official pip website](https://pip.pypa.io/en/stable/installation/).


4. Clone the repository:
    ```sh
    git clone <repository-url>
    ```

5. Navigate to the `responsible-ai-backend` directory:
    ```sh
    cd responsible-ai-backend
    ```

6. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

7. Activate the virtual environment:
    - On Windows:
        ```sh
        .\venv\Scripts\activate
         ```

    - On Linux/Mac:
        ```sh
        source venv/bin/activate
        ```

8. Go to the `requirement` directory where the `requirements.txt` file is present and install the requirements:
    ```sh
    pip install -r requirements.txt
    ```
## Set Configuration Variables

After installing all the required packages, configure the variables necessary to run the APIs.

1. Navigate to the `backend` directory:
    ```sh
    cd ..
    cd src/rai_backend
    ```

2. Locate the `.env` file in the project directory. This file contains several configuration keys. Make sure to fill in the mandatory fields as shown below (sample values are provided for reference):

    ```sh
    DB_NAME = "${dbname}"                     # [Mandatory] DB_NAME = "raibackend"
    DB_USERNAME = "${username}"               # [Optional]
    DB_PWD = "${password}"                    # [Optional]
    DB_IP = "${ipaddress}"                    # [Optional]
    DB_PORT = "${port}"                       # [Optional]
    DB_TYPE ="${dbtype}"                      # [Mandatory] DB_TYPE = "mongo"
    MONGO_PATH = "mongodb://${DB_USERNAME}:${DB_PWD}@${DB_IP}:${DB_PORT}/"    # [Mandatory] MONGO_PATH = "mongodb://localhost:27017/"
    COSMOS_PATH = "${cosmos_path}"            # [Optional]
    SECRET_KEY = "${your_secret_key}"         # [Mandatory] SECRET_KEY= "your_secret_key" eg : you can give some random value like rai123
    ALGORITHM = "${your_algorithm}"           # [Mandatory] ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = "${minutes}" # [Mandatory] ACCESS_TOKEN_EXPIRE_MINUTES = 30 
    TOKEN_URL = "${your_token_url}"            # [Mandatory] TOKEN_URL =  "/token"
    AUTHENTICATE_TELEMETRY_URL = "${authenticateTelemetryUrl}"            # [Optional]
    TELEMETRY_FLAG = "${telemetryFlag}"                                    # [Optional] False if do not want to connect with telemetry 

    ```
    ```sh
    allow_origin= "${allow_origin}"     # ALLOW_ORIGINS ="*" 
    allow_methods = "${allow_methods}"   # ALLOW_METHODS ="*"        
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

3. PORT_NO : Use the Port No that is configured in `.env` file.

   Open the following URL in your browser:
    `http://localhost:<portno>/v1/rai/backend/docs`

    User can also change the port which mentioned in main.py file
    
## Account API Details
    The /account API will not work from the swagger docs. It is used to verify and the JWT token is sent from the front end and then the api will be authorized. 

## ResetCount API Details
    The resetcount API will be removed in the next release from the swagger docs.
  
## License

The source code for the project is licensed under the MIT license, which you can find in the [LICENSE.md](License.md) file.

## Contact

If you have more questions or need further insights, feel free to Connect with us @ infosysraitoolkit@infosys.com

