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
load_dotenv()

# api_key = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=api_key)


st.title('Aurora Testing')
# model = genai.GenerativeModel('gemini-1.5-flash')
# config = genai.types.GenerationConfig(temperature=0.7, max_output_tokens=4000, top_p=0.95, top_k=64)
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

# Testing recommendation feature
# Testing Completed

# Testing vision feature
# Testing Completed

# Testing GSheets Connection
# Testing Completed

# # User contact us feature
# st.header('📧Contact Us', divider='rainbow')
# st.subheader("Have a query or feedback? Reach out to us!", divider='rainbow')

# # Establish a connection to Google Sheets
# conn = st.connection("gsheets", type=GSheetsConnection)

# feedback_df = conn.read(worksheet="Feedback Data", ttl=5)
# query_df = conn.read(worksheet="Query Data", ttl=5)
# # st.dataframe(df)

# # Select the action
# action = st.selectbox("Select an action:", ["Query", "Feedback"])

# # Contact Form
# if action == "Feedback":
#         with st.form(key='contact_form'):
#             name = st.text_input("Name*", key='user_name')
#             email = st.text_input("Email*", key='user_email')
#             ratings = st.radio("Ratings", [1, 2, 3, 4, 5], key='rating')
#             message = st.text_area("Message*", key='message')
#             st.markdown("**Required*")
#             submit_button = st.form_submit_button("Submit")
#             if submit_button:
#                if not name or not email or not message:
#                   st.error("Please fill in the required fields.")
#                   st.stop()
#                elif feedback_df["Email"].str.contains(email).any():
#                   st.warning("You have already submitted a feedback.")
#                   st.stop()
#                else:
#                   user_feedback_data = pd.DataFrame({
#                       "Name": name,
#                       "Email": email,
#                       "Ratings": ratings,
#                       "Message": message
#                   }, index=[0])

#                   updated_df = pd.concat([feedback_df, user_feedback_data], ignore_index=True)

#                   # Update Google Sheets with new data
#                   conn.update(worksheet="Feedback Data", data=updated_df)
#                   st.success("Feedback submitted successfully.")
 
# if action == "Query":
#         with st.form(key='contact_form'):
#             name = st.text_input("Name*", key="user_name")
#             email = st.text_input("Email*", key="user_email")
#             subject = st.text_input("Subject*", key="subject")
#             upload_img = st.file_uploader("Upload Image (Optional)", type=["jpg", "jpeg", "png"], key="upload_img")
#             message = st.text_area("Message*", placeholder="Explain your query in detail.", key="message")
#             check_box = st.checkbox("I agree to be contacted for further details.", key="check_box")
#             st.markdown("**Required*")
#             submit_button = st.form_submit_button("Submit")
#             if submit_button:
#                 if not name or not email or not subject or not message:
#                     st.error("Please fill in the required fields.")
#                     st.stop()
#                 elif not check_box:
#                     st.error("Please agree to be contacted for further details.")
#                     st.stop()
#                 else:
#                     query_df["Email"] = query_df["Email"].astype(str)
                    
#                     if query_df["Email"].str.contains(email).any():
#                       st.warning("You have already submitted a query.")
#                       st.stop()
#                     else:
#                       user_query_data = pd.DataFrame({
#                           "Name": name,
#                           "Email": email,
#                           "Subject": subject,
#                           "Upload Image": upload_img,
#                           "Message": message
#                       }, index=[0])

#                       updated_df = pd.concat([query_df, user_query_data], ignore_index=True)

#                       # Update Google Sheets with new data
#                       conn.update(worksheet="Query Data", data=updated_df)
#                       st.success("Query submitted successfully.")