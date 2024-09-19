import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
# from pii_inference.utils import PiiNERPipeline
from datasets import Dataset
# from pii_redaction.utils import get_replacements, redact_pii_batch
from privacy.util.code_detect.ner.pii_redaction.utils import get_replacements, redact_pii_batch
import json
import re
import os
from privacy.util.code_detect.ner.pii_inference.utils.pipeline import PiiNERPipeline
class codeNer:
    def codeFile(code, filename,model,tokenizer):
        # Specify the path to your local model and input code file
        # model_path = "pii_inference/nermodel"
        # code_file_path = "1.txt"
        # redacted_code_file_path = "1_redacted.txt"
        # Load the model and tokenizers
        # model = AutoModelForTokenClassification.from_pretrained(model1)
        # tokenizer = AutoTokenizer.from_pretrained(tokenizer1)
        
        # If code is a bytes object, decode it to a string
        if isinstance(code, bytes):
            code = code.decode('utf-8')
        print(code, "CODE")
        # Create the NER pipeline
        pipeline = PiiNERPipeline(
            model,
            tokenizer=tokenizer,
            batch_size=1024,
            window_size=512,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_workers=1,
            id_to_label=model.config.id2label,
            window_overlap=False,
            bf16=True
        )

        # # Read the input code file
        # with open(code_file_path, "r",encoding="utf-8") as file:
        #     code = file.read()

        # Split the code into sentences
        # sentences = code.split(". ")
        sentences = re.split('(?<=\. )(?!$|[a-z])', code)
        # sentences = code
        print(sentences, "############SENTENCES#########")

        # Create an id list
        ids = list(range(len(sentences)))

        # Create a Dataset object from the sentences
        dataset = Dataset.from_dict({"content": sentences, "id": ids})

        replacements = get_replacements()

        # Process the sentences with the NER pipeline
        result = pipeline(dataset)

        # Convert the generator to a list and print the results
        results = list(result)
        print(results, "RESULT")

        # # Check the structure of each result
        # for res in results:
        #     print(res)

        # Prepare the examples for redaction
        examples = {
            "content": [res["content"] for res in results],
            "entities": [res["entities"] for res in results]
        }

        # Print examples for debugging
        # print(examples, "EXAMPLES")

        # Redact PII in the batch of examples
        redacted_results = redact_pii_batch(examples, replacements)
        print(redacted_results, "redacted_code_parts")
        # Extract the redacted code from the results
        redacted_code_parts = redacted_results["new_content"]
        print(redacted_code_parts, "redacted_code_parts")
        # redacted_code = "".join(redacted_code_parts)
        # print(redacted_code, "redacted_code")
        # print(redacted_code1, "redacted_code1")
        # Save the redacted code into a new file
        # Generate the output file name based on the input file name
        output_code_file = os.path.splitext(filename)[0] + "_redacted" + os.path.splitext(filename)[1]
        try:
            # with open(output_code_file, "w") as file:
            #     for part in redacted_code_parts:
            #         file.write(part)
            #         print(part, "part")
            #     print(part, "part")
            with open(output_code_file, "w") as file:
                # Join all the parts into a single string
                content = ''.join(redacted_code_parts)
                # Write the content to the file
                file.write(content)
                print(content, "content")    
            # with open(output_code_file, "r") as file:
            #     content = file.read()
            # print(content,"content from function")
                # file.write(redacted_code_parts)
        except Exception as e:
            print(f"An error occurred while writing to the file: {e}")
            
        return content, output_code_file
    
    def codeText(code, model, tokenizer):
        # If code is a bytes object, decode it to a string
        if isinstance(code, bytes):
            code = code.decode('utf-8')
        print(code, "CODE")
        # Create the NER pipeline
        pipeline = PiiNERPipeline(
            model,
            tokenizer=tokenizer,
            batch_size=1024,
            window_size=512,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_workers=1,
            id_to_label=model.config.id2label,
            window_overlap=False,
            bf16=True
        )

        # Split the code into sentences
        sentences = re.split('(?<=\. )(?!$|[a-z])', code)
        print(sentences, "############SENTENCES#########")

        # Create an id list
        ids = list(range(len(sentences)))

        # Create a Dataset object from the sentences
        dataset = Dataset.from_dict({"content": sentences, "id": ids})

        replacements = get_replacements()

        # Process the sentences with the NER pipeline
        result = pipeline(dataset)

        # Convert the generator to a list and print the results
        results = list(result)
        print(results, "RESULT")

        # Prepare the examples for redaction
        examples = {
            "content": [res["content"] for res in results],
            "entities": [res["entities"] for res in results]
        }

        # Redact PII in the batch of examples
        redacted_results = redact_pii_batch(examples, replacements)
        print(redacted_results, "redacted_code_parts")
        # Extract the redacted code from the results
        redacted_code_parts = redacted_results["new_content"]
        print(redacted_code_parts, "redacted_code_parts")

        # Join all the parts into a single string
        redacted_code = ''.join(redacted_code_parts)
        print(redacted_code, "redacted_code")

        return redacted_code

# if __name__ == "__main__":
#     main()