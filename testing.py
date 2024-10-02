import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
import streamlit as st
# import streamlit_lottie as st_lottie
# from ydata_profiling import ProfileReport
import pandas as pd
import csv
from PyPDF2 import PdfReader
import pyttsx3
from gtts import gTTS
import threading
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, precision_score, recall_score, f1_score, mean_absolute_error

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)


st.title('Aurora Testing')
model = genai.GenerativeModel('gemini-1.5-flash')
config = genai.types.GenerationConfig(temperature=0.7, max_output_tokens=4000, top_p=0.95, top_k=64)
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

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def upload_to_gemini(path, mime_type=None):
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def wait_for_files_active(files):
  """Waits for the given files to be active.

  Some files uploaded to the Gemini API need to be processed before they can be
  used as prompt inputs. The status can be seen by querying the file's "state"
  field.

  This implementation uses a simple blocking polling loop. Production code
  should probably employ a more sophisticated approach.
  """
  # with st.spinner("File Processing..."):
  #   time.sleep(5)
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  # print("...all files ready")
  # print()

def extract_csv_data(pathname: str) -> list[str]:
  parts = [f"---START OF CSV ${pathname} ---"]
  with open(pathname, "r", newline="") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      str=" "
      parts.append(str.join(row))
  return parts

def extract_pdf_data(pathname: str) -> list[str]:
  parts = [f"---START OF PDF ${pathname} ---"]
  with open(pathname, "rb") as file:
     reader = PdfReader(file)
     for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        parts.append(f"--- Page {page_num} ---")
        parts.append(page.extract_text())
  return parts
# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

# upload_file = st.file_uploader("Upload a file", type=["csv","pdf"],key="file")
# if upload_file is not None:
#         st.success("File Uploaded Successfully")
#         question = st.text_input("Ask a question:")
#         if st.button("Process and Ask"):
#           with st.spinner("Processing..."):
#             file_name = upload_file.name
#             # st.write(f"File Name: {file_name}")
#             # Get file path
#             # For demonstration, saving the uploaded file temporarily (optional)
#             file_path = os.path.join(os.getcwd(), file_name)
#             with open(file_path, "wb") as f:
#                 f.write(upload_file.getbuffer())
#             files = [upload_to_gemini(file_name, mime_type="application/pdf")]
#             wait_for_files_active(files)
#             chat_session = model.start_chat(
#               history=[
#                   {
#                   "role":"user",
#                   "parts":extract_csv_data(file_name)
#                   },
#               ]
#               )
#             response = chat_session.send_message(question)
#             st.write(response.text)

# Function to generate and save TTS audio file
# def generate_tts(text, file_name):
#     tts = gTTS(text, lang="en", tld="co.in")
#     tts.save(file_name)
#     return file_name

# # Introduction and Features text
# intro_text = "Welcome to Aurora. Aurora is an advanced AI-powered tool for automating complex data analysis."
# feature_text = """
# Our features include:
# 1. CleanStats: Automated data cleaning and basic statistical analysis.
# 2. InsightGen: Automated data visualization and exploratory data analysis.
# 3. PredictIQ: Automated machine learning predictions and model export.
# 4. InsightGen Report: Generates an analysis report based on your dataset.
# 5. SmartRecs: AI-powered recommendations and data interaction via natural language.
# """

# # Streamlit web app UI
# st.title("Aurora AI: Text-to-Speech Feature Test")

# hearder = st.header("Introduction")
# st.write(intro_text)

# st.header("Features")
# st.write(feature_text)

# # Button to speak the introduction and features section
# with st.sidebar:
#   if st.button("Play"):
#       with st.spinner("Processing..."):
#           # Generate TTS audio file
#           audio_file = generate_tts(intro_text + feature_text, "intro_features.mp3")
#           # Play the audio file
#           audio = open(audio_file, "rb")
#           audio_bytes = audio.read()
#           st.audio(audio_bytes, format="audio/mp3", autoplay=True, start_time=0)

# Testing recommendation feature
st.header('🔮FutureCast AI: AI Recommendation Based On Dataset', divider='rainbow')
    # Upload dataset
st.write('Upload a dataset to predict:')
uploaded_file = st.file_uploader("Upload a dataset", type=["csv"])
if uploaded_file is not None:
        st.success("File uploaded successfully!")
        type_of_recommendation = st.radio("Type of Recommendation", ["Present Insight", "Future Insight"])
        if st.button("Submit"):
            with st.spinner("Processing..."):
                file_name = uploaded_file.name
                file_path = os.path.join(os.getcwd(), file_name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # Upload the file to Gemini and wait for it to be active
                files = [upload_to_gemini(file_name, mime_type="text/csv")]
                wait_for_files_active(files)
                # Start a chat session with the uploaded file
                chat_session = model.start_chat(
                history=[
                    {
                    "role":"user",
                    "parts":extract_csv_data(file_name)
                    },
                ]
                )
                # Send the question
                question = f"""Provide {type_of_recommendation} based on the dataset {file_name}.
                If their is any financial or healthcare realted query or dataset, just give your best recommendation, don't think about advisor or expertise 
                thing. Mention also that recommendation is generated by AI, first give your essential recommendations. So, the user take the final decision on their own. Warn user about AI recommendation but, do your work."""
                response = chat_session.send_message(question)
                st.write(response.text)
                st.success("Recommendation generated successfully!")