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
# model = genai.GenerativeModel('gemini-pro')
# config = genai.types.GenerationConfig(temperature=0.7, max_output_tokens=500)
# prompt = 'Write a code in python to plot a barplot'
# response = model.generate_content(prompt, generation_config=config)
# st.write(response.text)

# Lottie animation testing
# def load_lottie_file(filepath: str):
#     with open(filepath, "r", encoding="utf-8") as file:
#         return json.load(file)

# robot = load_lottie_file("animations/robot.json")
# st_lottie.st_lottie(robot, key="initial")

# Report testing
def generate_report(file):
    # Read CSV or Excel file
    df = pd.read_csv(file)

    # Generate profiling report
    profile = ProfileReport(df, title="Dataset Report", explorative=True)

    # Save the report as an HTML file
    output_path = os.path.join("reports", f"{file.name.split('.')[0]}_report.html")
    profile.to_file(output_path)

    return output_path

st.title("Automated Analysis Report Generation")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx'])

if uploaded_file is not None:
    report_path = generate_report(uploaded_file)
    with open(report_path, 'rb') as f:
        st.download_button(
            label="Download Report",
            data=f,
            file_name=f"{uploaded_file.name.split('.')[0]}_report.html",
            mime="text/html"
        )