import google.generativeai as genai
import os
import time
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

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

upload_file = st.file_uploader("Upload a file", type=["csv", "txt"])
question = st.text_input("Ask a question")

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
  st.write("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  st.write("...all files ready")
  print()

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

# TODO Make these files available on the local file system
# You may need to update the file paths
if upload_file is not None:
    filename = upload_file.name
    files = [
    upload_to_gemini(filename, mime_type="text/csv"),
    ]

    # Some files have a processing delay. Wait for them to be ready.
    wait_for_files_active(files)

    chat_session = model.start_chat(
    history=[
        {
        "role": "user",
        "parts": [
            files[0],
            "What content is in this file?",
        ],
        },
        {
        "role": "model",
        "parts": [
            "The file \"Iris.csv\" contains data about 150 iris flowers. Each row represents a single flower, and the columns describe its characteristics:\n\n* **Id**: A unique identifier for each flower.\n* **SepalLengthCm**: The sepal length in centimeters.\n* **SepalWidthCm**: The sepal width in centimeters.\n* **PetalLengthCm**: The petal length in centimeters.\n* **PetalWidthCm**: The petal width in centimeters.\n* **Species**: The species of iris, which can be one of three: Iris-setosa, Iris-versicolor, or Iris-virginica. \n\nThis dataset is a classic example of a machine learning dataset, often used to demonstrate classification algorithms.  \n",
        ],
        },
    ]
    )

    response = chat_session.send_message(question)

    st.write(response.text)