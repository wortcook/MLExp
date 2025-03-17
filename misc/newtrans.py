
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#suppress all warnings
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME='google-t5/t5-base'

# Create a Tokenizer for Machine Translation
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Instantiate a Seq2Seq model from the specified pretrained model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_translation(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Perform the translation and decode the output
    outputs = model.generate(**inputs)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded_output

ENGLISH_TO_FRENCH = 'translate English to French: '
ENGLISH_TO_GERMAN = 'translate English to German: '
FRENCH_TO_ENGLISH = 'translate French to English: '
GERMAN_TO_ENGLISH = 'translate German to English: '

# Define the input text
text = "The large, beautiful house is wonderful."
print(text)
text_01 = generate_translation(ENGLISH_TO_FRENCH + text)
print(text_01)
text_02 = generate_translation(FRENCH_TO_ENGLISH + text_01)
print(text_02)
text_03 = generate_translation(ENGLISH_TO_GERMAN + text)
print(text_03)
text_04 = generate_translation(GERMAN_TO_ENGLISH + text_03)
print(text_04)
