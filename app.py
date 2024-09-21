import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from deep_translator import GoogleTranslator
import torch
device = 0 if torch.cuda.is_available() else -1

# Load Arabic summarization model and tokenizer
model_ckpt = "Arabic-Text_summazier"  # Replace with your model checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, legacy=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

def extract_summary(generated_text):
    prefix_to_remove = "Summary :"
    prefix_to_remove2 = "Summary:"
    generated_text = str(generated_text)
    cleaned_text = generated_text[len(prefix_to_remove):] if generated_text.startswith(prefix_to_remove) else generated_text
    cleaned_text2 = cleaned_text[len(prefix_to_remove2):] if cleaned_text.startswith(prefix_to_remove2) else cleaned_text
    return cleaned_text2

def summarize_text(prompt):
    generator = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=device)
    result = generator(prompt, max_length=800)
    return extract_summary(result[0]['generated_text'])

# Streamlit UI
st.title("Arabic Text Summarization and Translation")

# Input box for Arabic text
input_text = st.text_area("Enter Arabic text here:")


if st.button("Generate Summary and Translate"):
    if input_text:
        translator = GoogleTranslator(source='ar', target='en')
        translation = translator.translate(input_text)

        st.subheader("Input Translation:")
        st.write(translation)

    if input_text:
        st.subheader("")
        # Generate summary
        summary = summarize_text(input_text)
        
        # Translate summary to English
        translator = GoogleTranslator(source='ar', target='en')
        translation = translator.translate(summary)
        
        # Display results
        st.subheader("Generated Summary:")
        st.write(summary)
        
        st.subheader("Translation in English:")
        st.write(translation)
    else:
        st.warning("Please enter some Arabic text to summarize and translate.")
