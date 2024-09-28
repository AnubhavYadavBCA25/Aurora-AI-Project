import google.generativeai as genai
import os
from dotenv import load_dotenv
import streamlit as st
# import streamlit_lottie as st_lottie
# from ydata_profiling import ProfileReport
import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)


st.title('Aurora Testing')
model = genai.GenerativeModel('gemini-pro')
config = genai.types.GenerationConfig(temperature=0.7, max_output_tokens=1000)
# prompt = 'Write a code in python to plot a barplot'
# response = model.generate_content(prompt, generation_config=config)
# st.write(response.text)

# Lottie animation testing
# def load_lottie_file(filepath: str):
#     with open(filepath, "r", encoding="utf-8") as file:
#         return json.load(file)

# robot = load_lottie_file("animations/robot.json")
# st_lottie.st_lottie(robot, key="initial")

# Report Generation Testing
# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#     filename = uploaded_file.name
#     df = pd.read_csv(uploaded_file)
#     summary = df.describe().transpose().to_string()
#     st.dataframe(df)
#     if df is not None:
#         prompt = f'Generate a text report for the dataset {filename} and summary of the dataset is {summary}.'
#         response = model.generate_content(prompt, generation_config=config)
#         st.write(response.text)
    
# # Profile Report Testing
# if st.button('Generate Profile Report'):
#     profile = ProfileReport(df, title="Pandas Profiling Report")
#     profile.to_file("output.html")
#     st.write("Profile Report generated successfully")

# AI Recommender/Dataset Chat Feature Testing
uploaded_file = st.file_uploader("Choose a file")
question = st.text_input("Ask a question about the dataset")
if uploaded_file and question is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

    # Generate a summary of the dataset
    dataset_summary = f"""
    Columns: {', '.join(df.columns)}
    Number of rows: {len(df)}
    Basic statistics: {df.describe().to_dict()}
    """

    # Prompt template
    template = """
    You are a dataset assistant. Based on the following dataset summary, answer the user's question:
    Dataset Summary: {dataset_summary}

    User Question: {question}
    """

    prompt = PromptTemplate(template=template, input_variables=["dataset_summary"])

    # LLMChain for interacting with the dataset
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    response = llm_chain.run(template)

    st.write(response.text)