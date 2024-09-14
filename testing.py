import google.generativeai as genai
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

def models():
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)

st.title('Generative AI Testing')
model = genai.GenerativeModel('gemini-pro')
config = genai.types.GenerationConfig(temperature=0.7, max_output_tokens=500)
prompt = 'Write a code in python to plot a barplot'
response = model.generate_content(prompt, generation_config=config)
st.write(response.text)