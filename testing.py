import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
import streamlit as st
# import streamlit_lottie as st_lottie
# from ydata_profiling import ProfileReport
import pandas as pd
import csv
from pathlib import Path

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

upload_file = st.file_uploader("Upload a file", type=["csv", "txt"],key="file")
if upload_file is not None:
        st.success("File Uploaded Successfully")
        question = st.text_input("Ask a question:")
        if st.button("Process and Ask"):
          file_name = upload_file.name
          # Get file path
          # For demonstration, saving the uploaded file temporarily (optional)
          file_path = os.path.join(os.getcwd(), file_name)
          with open(file_path, "wb") as f:
              f.write(upload_file.getbuffer())
          files = [upload_to_gemini(file_name, mime_type="text/csv")]
          wait_for_files_active(files)
          chat_session = model.start_chat(
          history=[
              {
              "role":"user",
              "parts":extract_csv_data(file_name)
              },
          ]
          )
          response = chat_session.send_message(question)
          st.write(response.text)
# if upload_file is not None:
#         filename = upload_file.name
#         file_path = os.path.join(os.getcwd(), filename)
#         with open(file_path, "wb") as f:
#             f.write(upload_file.getbuffer())
#         if st.button("Submit and Process"):
#            with st.spinner("Processing..."):
#               files = [
#                   upload_to_gemini(file_path, mime_type="text/csv"),
#                   ]
#               wait_for_files_active(files)
#               st.success("File Processed Successfully")
#               question = st.text_input("Ask a question", "Tell me the data of the row where Id is 100.")
#               if st.button("Ask"):
#                 chat_session = model.start_chat(
#                   history=[
#                       {
#                           "role": "user",
#                           "parts": extract_csv_data(filename),
#                       },
#                   ]
#               )

#                 response = chat_session.send_message(question)

#                 st.write(response.text)
#                 st.success("Done")