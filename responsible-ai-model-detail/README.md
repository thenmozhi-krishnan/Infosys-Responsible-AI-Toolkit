# responsible-ai-model-detail


## Introduction
The `responsible-ai-model-detail` repository is used for uploading model and data files. After successfully uploading, we will get their IDs, which are needed for generating a batch ID based on our tenant, used for performing our tenant's work.


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

5. Navigate to the `responsible-ai-model-detail` directory:
    ```sh
    cd responsible-ai-model-detail
    ```

6. Navigate to `src` directory:
    ```sh
    cd workbench\src
    ```

7. Go to the `requirements` directory where the `requirement.txt` file is present
    - If On Windows:
        Do the needful as written in the end section of file
    - Now, install the requirements:
        ```sh
        pip install -r ../requirements/requirement.txt
        ```

## Running the Application
Once we have completed all the aforementioned steps, we can start the service.

1. Run `main.py` file:
    ```sh
    python main.py
    ```

2. Open the following URL in your browser:
    [http://localhost:80/v1/workbench/docs](http://localhost:80/v1/workbench/docs)


## License
The source code for the project is licensed under MIT license, which you can find in the [LICENSE.md](LICENSE.md) file.


## Contact
If you have more questions or need further insights, feel free to Connect with us @ Infosysraitoolkit@infosys.com
