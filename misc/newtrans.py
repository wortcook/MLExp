
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME='google-t5/t5-11b'

# Create a Tokenizer for Machine Translation
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Instantiate a Seq2Seq model from the specified pretrained model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Define the input text
text = "translate English to French: The house is wonderful."

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Perform the translation and decode the output
outputs = model.generate(**inputs)
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the translated text
print(decoded_output)
