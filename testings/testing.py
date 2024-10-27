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

# Testing streamlit form on each feature
# Feature 1: Data cleaning and stats analysis
# Feature 1 Testing Completed

# Feature 2: AutoVisualization
# Feature 2 Testing Completed
# st.header('📈AutoViz: Data Visualization & EDA', divider='rainbow')
#     # Upload dataset
# with st.form(key='data_visualization_form'):
#         st.write("Upload a dataset to generate visualizations.")
#         uploaded_file = st.file_uploader("Choose a file")
#         # Select the visualization type
#         visualization_type = st.selectbox("Select the visualization type", ["Barplot", "Histogram", "Boxplot", "Scatterplot", "Lineplot"])
#         user_input = st.text_input("Enter the columns for visualization separated by 'and', Example: column1 and column2")
#         submitted = st.form_submit_button("Submit")
#         if submitted:
#             st.success("File and visualization type submitted successfully!")

# with st.spinner("Generating Visualization..."):
#   if uploaded_file and visualization_type and user_input is not None:
#       file_name = uploaded_file.name
#       file_path = os.path.join("uploads", file_name)
#       df = load_file(uploaded_file)
#       df = df_cleaning(df)
#       df_sample = str(df.head())
#       columns = user_input
#       st.subheader(f"{visualization_type} Visualization for the dataset '{file_name}' for the columns {columns}:")
#       predefined_prompt = f"""Write a python code to plot a {visualization_type} using Matplotlib or Seaborn Library. Name of the dataset is {file_name}.
#       Plot for the dataset columns {columns}. Here's the sample of dataset {df_sample}. Set xticks rotation 90 degree. 
#       Set title in each plot. Add tight layout in necessary plots. Don't right the explanation, just write the code."""
#       response = model.generate_content(predefined_prompt, generation_config=config)
#       generated_code = response.text
#       generated_code = generated_code.replace("```python", "").replace("```", "").strip()
#       # Modify the code to insert the actual file path into pd.read_csv()
#       if "pd.read_csv" in generated_code:
#           generated_code = generated_code.replace("pd.read_csv()", f'pd.read_csv(r"{file_path}")')
#       elif "pd.read_excel" in generated_code:
#           generated_code = generated_code.replace("pd.read_excel()", f'pd.read_excel(r"{file_path}")')
#       st.code(generated_code, language='python')
#       try:
#           exec(generated_code)
#           st.pyplot(plt.gcf())
#       except Exception as e:
#           st.error(e)
#       st.success("Visualization generated successfully!")