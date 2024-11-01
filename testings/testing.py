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
from streamlit_gsheets import GSheetsConnection
from PIL import Image
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)


st.title('Aurora Testing')
model = genai.GenerativeModel('gemini-1.5-flash')
config = genai.types.GenerationConfig(temperature=0.7, max_output_tokens=4000, top_p=0.95, top_k=64)
# prompt = 'Write a code in python to plot a barplot'
# response = model.generate_content(prompt, generation_config=config)
# st.write(response.text)

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

# Functions
# Function for loading csv format file
@st.cache_data
def load_csv_format(file):
        df = pd.read_csv(file)
        return df

# Function for loading xlsx format file
@st.cache_data
def load_xlsx_format(file):
        df = pd.read_excel(file)
        return df

# Function for loading file based on its format
@st.cache_data
def load_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
            return load_csv_format(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
            return load_xlsx_format(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or XLSX file.")

@st.cache_data
# Function for data cleaning
def df_cleaning(df):
    df = df.drop_duplicates()

    # Impute missing values
    # Seperate numerical and object columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    object_columns = df.select_dtypes(include=['object']).columns

    # Impute missing values for numerical columns and object columns
    numerical_imputer = SimpleImputer(strategy='mean')
    df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])

    object_imputer = SimpleImputer(strategy='most_frequent')
    df[object_columns] = object_imputer.fit_transform(df[object_columns])
    return df

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

@st.cache_data
def generate_report(df,file):
    # Generate profiling report
    profile = ProfileReport(df, title="Dataset Report", explorative=True)

    # Save the report as an HTML file
    output_path = os.path.join("reports", f"{file.name.split('.')[0]}_report.html")
    profile.to_file(output_path)
    return output_path
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

# Testing recommendation feature
# Testing Completed

# Testing vision feature
# Testing Completed

# Testing GSheets Connection
# Testing Completed

# Testing streamlit form and modifications on each feature
# Feature 1: Data cleaning and stats analysis (some modifications remaining)
# Feature 1 Testing Completed

# Feature 2: AutoVisualization (some modifications remaining)
# Feature 2 Testing Completed

# Feature 3: AI Recommendation
# Feature 3 Testing Completed

# Feature 4: Report Generation
# Feature 4 Testing Completed

# Feature 5: AI CSV File ChatBot
# Feature 5 Testing Completed

# Feature 6: Image Analysis
st.header('👁️VisionFusion: AI-Powered Image Analysis ', divider='rainbow')
with st.form(key='img_analysis'):
  st.write('Upload an image to analyze:')
  uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
  user_query = st.text_input("Ask a query:")
  submitted = st.form_submit_button("Submit")
  if submitted:
      if not uploaded_image or not user_query:
          st.error("Please upload both required fields to proceed!")
          st.stop()
      else:
          st.success("Image and Query submitted successfully!")

with st.spinner("Processing..."):
  if uploaded_image and user_query is not None:
          # Show uploaded image
          st.subheader('Uploaded Image')
          st.image(uploaded_image, caption="Uploaded Image", use_column_width=False)
          file_name = uploaded_image.name
          st.subheader(f"'{file_name}' Image Analysis:")
          st.divider()
          image = Image.open(uploaded_image)
          prompt = f"Analyze the image and provide a detailed description of the image. {user_query}"
          response = model.generate_content([prompt,image], stream=True)
          response.resolve()
          st.write(response.text)
          st.success("Image analyzed successfully!")
  else:
      st.warning("Please upload a dataset and ask a question to proceed!")