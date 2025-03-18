
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#suppress all warnings
import warnings
warnings.filterwarnings("ignore")

# MODEL_NAME='google-t5/t5-11b'
# MODEL_NAME='facebook/m2m100_418M'
MODEL_NAME='facebook/m2m100_1.2B'

# Create a Tokenizer for Machine Translation
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Instantiate a Seq2Seq model from the specified pretrained model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_translation(text, model, tokenizer, source_lang="en", target_lang="fr"):

    tokenizer.src_lang = source_lang

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", )

    # Perform the translation and decode the output
    outputs = model.generate(**inputs, max_length=1024, forced_bos_token_id=tokenizer.get_lang_id(target_lang))

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded_output

ENGLISH_TO_FRENCH = 'translate to French: '
ENGLISH_TO_GERMAN = 'translate to German: '
FRENCH_TO_ENGLISH = 'translate : '
GERMAN_TO_ENGLISH = 'translate : '

# Define the input text
text = "This universe, in particular, is Universe Today, a space news website a good quarter-century old published by Fraser Cain. Over that time, it has sprouted podcasts and a YouTube channel with nearly half a million subscribers. It has a team of expert writers and production staff, runs interviews with actual researchers about all manner of space science and events, and is highly regarded. In short, it's doing the sort of science journalism that seems most under threat in today's turgid political and anti-factual times. It is a very good thing."
print("Original text: "+text)
text_01 = generate_translation(text, model, tokenizer, source_lang="en", target_lang="fr")
print("French translation: "+text_01)
text_02 = generate_translation(text_01, model, tokenizer, source_lang="fr", target_lang="en")
print("Back to English: "+text_02)
text_03 = generate_translation(text_02, model, tokenizer, source_lang="en", target_lang="de")
print("German translation: "+text_03)
text_04 = generate_translation(text_03, model, tokenizer, source_lang="de", target_lang="en")
print("Back to English: "+text_04)
