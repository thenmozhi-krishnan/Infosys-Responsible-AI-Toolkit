'''
Copyright 2024-2025 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect
from config.logger import CustomLogger

log = CustomLogger()

class ModelBasedTranslate:
    def __init__(self):
        try:
            self.model_name = "facebook/m2m100_418M"
            self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)
            self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
        except Exception as e:
            log.error(f"Failed to load model or tokenizer: {e}")
            raise

    def translate(self, text: str):
        try:
            # Detect language
            lang_code = detect(text)
            log.info(f"Detected language: {lang_code}")

            self.tokenizer.src_lang = lang_code
            encoded_text = self.tokenizer(text, return_tensors="pt")

            # Generate translation to English
            generated_tokens = self.model.generate(**encoded_text, forced_bos_token_id=self.tokenizer.get_lang_id("en"))
            
            # Decode tokens to text
            translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            log.info(f"Translated text: {translated_text}")
            
            return translated_text, lang_code

        except Exception as e:
            log.error(f"Exception during translation: {e}")
            # Fallback or error handling
            return text, "en" # Assume english on failure
