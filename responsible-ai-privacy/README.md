# responsible-ai-privacy

## Introduction
Privacy is application which detects and masks any PII data present in Unstructured, Text, Image, DICOM, Video and returns the processed data.
 
## Requirements
1. Python 3.9 - 3.11
2. VSCode
# Models
 1. En_core_wb_lg Model Download and place it lib folder:
     link:[https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl](https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl)
     docs: https://spacy.io/models/en#en_core_web_lg
 2. StarPII Model for code moderation: https://huggingface.co/bigcode/starpii
    Download the model files, paste the model creating folder named as nermodel inside, src\privacy\util\code_detect\ner\pii_inference\nermodel
## Steps to run this module :
1. Clone this repository in vscode
2. Create a virtual environment for python using cmd -
   `python -m venv <env-name>`
3. Activate the python virtual environment and install all the dependencies in requirement.txt file of the     cloned repository -
   `pip install -r path/to/requirements.txt`
4. Open .env file in vscode and configure the entries in it
5. In the virtual environment go to src folder of cloned repository and run below command to run the module-
   `py main.py`


## License
The source code for the project is licensed under the MIT license, which you can find in the [LICENSE.txt](LICENSE.txt) file.

## Contact
If you have more questions or need further insights please feel free to connect with us @ Infosysraitoolkit@infosys.com
