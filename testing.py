import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import streamlit as st
import streamlit_lottie as st_lottie
from ydata_profiling import ProfileReport
import pandas as pd

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

def models():
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)

st.title('Aurora Testing')
model = genai.GenerativeModel('gemini-pro')
config = genai.types.GenerationConfig(temperature=0.7, max_output_tokens=1000)
prompt = 'Write a code in python to plot a barplot'
response = model.generate_content(prompt, generation_config=config)
st.write(response.text)

# Lottie animation testing
# def load_lottie_file(filepath: str):
#     with open(filepath, "r", encoding="utf-8") as file:
#         return json.load(file)

# robot = load_lottie_file("animations/robot.json")
# st_lottie.st_lottie(robot, key="initial")

# Report Generation Testing
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    filename = uploaded_file.name
    df = pd.read_csv(uploaded_file)
    summary = df.describe().transpose().to_string()
    st.dataframe(df)
    if df is not None:
        prompt = f'Generate a text report for the dataset {filename} and summary of the dataset is {summary}.'
        response = model.generate_content(prompt, generation_config=config)
        st.write(response.text)
    
# Profile Report Testing
if st.button('Generate Profile Report'):
    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file("output.html")
    st.write("Profile Report generated successfully")