import os
import time
import csv
import yaml
import json
from PIL import Image
from gtts import gTTS
import pandas as pd
import numpy as np
import streamlit as st
import streamlit_lottie as st_lottie
import matplotlib.pyplot as plt
import seaborn as sns
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import LoginError
from ydata_profiling import ProfileReport
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer

# Streamlit page configuration
st.set_page_config(
    page_title="Aurora AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

###################################################### User Authentication ######################################################

# Loading config file
with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialize session state for register page
if 'register' not in st.session_state:
    st.session_state['register'] = False

def show_login_form():
    # Creating the authenticator object
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['pre-authorized']
    )

    # Creating a login widget
    try:
        authenticator.login()
    except LoginError as e:
        st.error(e)

    if st.session_state["authentication_status"]:
        authenticator.logout('Logout',"sidebar")
        st.sidebar.write(f'Welcome **{st.session_state["name"]}**👋')
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')

    # Only show the "Register" button if the user is NOT logged in
    if st.session_state["authentication_status"] is None or st.session_state["authentication_status"] == False:
        st.write("---")
        if st.button("Register"):
            st.session_state['register'] = True  # Switch to register page

# Define function to show the register form
def show_register_form():
    st.write("## Register")
    new_username = st.text_input("Enter a new username")
    new_name = st.text_input("Enter your full name")
    new_password = st.text_input("Enter a new password", type="password")
    new_email = st.text_input("Enter your email")

    if st.button("Submit Registration"):
        if new_username and new_password and new_email:
            # Hash the new password
            hashed_password = stauth.Hasher([new_password]).generate()[0]

            # Update the config dictionary
            config['credentials']['usernames'][new_username] = {
                'name': new_name,
                'password': hashed_password,
                'email': new_email
            }
            # Save the updated credentials to the config.yaml file
            with open('config.yaml', 'w') as file:
                yaml.dump(config, file)

            st.success("User registered successfully! You can now log in.")
            st.session_state['register'] = False  # Go back to login page
        else:
            st.error("Please fill out all fields")

    # Add a "Back to Login" button to return to the login page
    if st.button("Back to Login"):
        st.session_state['register'] = False  # Return to login page

# Main section: Show either login or register form based on state
if st.session_state['register']:
    show_register_form()  # Show register form
else:
    show_login_form()  # Show login form

###################################################### AI Models ######################################################
# Gemini API
load_dotenv()
genai_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=genai_api_key)
model = genai.GenerativeModel('gemini-1.5-flash')
config = genai.types.GenerationConfig(temperature=1.0, max_output_tokens=1500, top_p=0.95, top_k=64)
config_for_chatbot = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain"
}
model_for_chatbot = genai.GenerativeModel(model_name='gemini-1.5-flash',generation_config=config_for_chatbot)

###################################################### Functions ######################################################
# Function for loading csv format file
def load_csv_format(file):
        df = pd.read_csv(file)
        return df
    
# Function for loading xlsx format file
def load_xlsx_format(file):
        df = pd.read_excel(file)
        return df

# Function for loading file based on its format
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

# Function for lottie file
def load_lottie_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)

# Function for generating report
def generate_report(df,file):
    # Generate profiling report
    profile = ProfileReport(df, title="Dataset Report", explorative=True)

    # Save the report as an HTML file
    output_path = os.path.join("reports", f"{file.name.split('.')[0]}_report.html")
    profile.to_file(output_path)
    return output_path

# Function for uploading file to Gemini
def upload_to_gemini(path, mime_type=None):
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
            if file.state.name != "ACTIVE":
                raise Exception(f"File {file.name} failed to process")

def extract_csv_data(pathname: str) -> list[str]:
  parts = [f"---START OF CSV ${pathname} ---"]
  with open(pathname, "r", newline="") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      str=" "
      parts.append(str.join(row))
  return parts

# Function to generate and save TTS audio file
def generate_tts(text, file_name):
    tts = gTTS(text, lang="en", tld="co.in")
    tts.save(file_name)
    return file_name

###################################################### Page 1: Introduction Page ######################################################
def introduction():
    st.header('🤖Aurora AI: AI Powered Data Analytics Tool', divider='rainbow')
    robot_file = load_lottie_file('animations_and_audios/robot.json')
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Introduction")
        intro_text = ('''
                    - **Aurora AI** is an AI-powered data analytics tool that provides a user-friendly interface for data cleaning, statistical analysis, data visualization, AI powered recommendations, automated data report generation and AI-powered dataset chatbot.
                    - It is designed to help users with little to no programming experience to perform complex data analysis tasks with ease.
                    - The tool is built using Python, Streamlit, and Gemini API for AI-powered content generation.
                    - It offers a wide range of features to help users explore, analyze, and gain insights from their data.
                    - The tool is equipped with AI models that can generate data visualizations, recommendation models, and automated data report generation based on user input.
                ''')
        st.markdown(intro_text)

    with right_column:
        st_lottie.st_lottie(robot_file, key='robot', height=450, width=450 ,loop=True)
    st.divider()

    left_column, right_column = st.columns(2)
    with right_column:
        st.subheader("Features:")
        feature_text = ('''
                    - **CleanStats:** A feature for data cleaning and statistical analysis. Where you can clean the data and get the basic statistics.
                    - **AutoViz:** A feature for data visualization and EDA. Where you can visualize the data using different plots.
                    - **FutureCast AI:** A feature for AI-based recommendations. Where you can get present and future insights based on the dataset.
                    - **InsightGen:** A feature for generating automated data reports. Where you can download the report in interactive HTML format.
                    - **SmartQuery:** A feature for AI-powered dataset chatbot. Where you can chat with the CSV data file and get the response.
                    - **VisionFusion:** A feature for AI-powered image analysis. Where you can analyze the images using AI models.''')
        st.markdown(feature_text)
    
    with left_column:
        features = load_lottie_file('animations_and_audios/features.json')
        st_lottie.st_lottie(features, key='features', height=350, width=350 ,loop=True)
    st.divider()

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader('Technology Stack:')
        st.markdown('''
            - **Python:** Core programming language used for data processing, machine learning, and backend logic.
            - **Streamlit:** Framework used to build the interactive web application and user interface.
            - **Pandas:** Library for efficient data manipulation, cleaning, and analysis.
            - **Matplotlib/Seaborn:** Libraries for generating data visualizations and plots.
            - **Gemini:** Google LLM Model used for AI-powered content generation.
            - **Streamlit-Authenticator:** For handling user authentication, login, and registration.
            - **.env:** For securely storing environment variables like API keys and sensitive data.
            - **gTTS:** Google Text-to-Speech library for generating audio files from text.''')
    with right_column:
        tech_used = load_lottie_file('animations_and_audios/tech_used.json')
        st_lottie.st_lottie(tech_used, key='tech', height=500, width=500 ,loop=True)

    with st.sidebar:
        if st.button("Introduce"):
            # with st.spinner("Processing..."):
            #     audio_file = generate_tts(intro_text + feature_text, "intro_features.mp3")
            #     audio = open(audio_file, "rb")
            #     audio_bytes = audio.read()
            st.audio('animations_and_audios/intro_features.mp3', format="audio/mp3", autoplay=True, start_time=0)            
    st.divider()

###################################################### Page 2: Statistical Analysis ######################################################
def statistical_analysis():
    st.header('🧹CleanStats: Cleaning & Statistical Analysis', divider='rainbow')
    # Upload dataset
    st.write('Upload a dataset for cleaning and statistical analysis:')
    uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx"])

    if uploaded_file is not None:
    # Load the file based on its format
        df = load_file(uploaded_file)
        st.success("File uploaded successfully!")
        if df is not None:
        # Remove duplicate rows
            df_cleaning(df)

            # Display dataset
            st.subheader("Dataset Preview:", divider='rainbow')
            st.dataframe(df)
            st.write("*Note: The dataset has been cleaned and missing values have been imputed. You can download the cleaned dataset for further analysis.*")
        
            # Basic statistics
            st.subheader("Basic Statistics:", divider='rainbow')
            st.write("For numerical columns:")
            st.write(df.describe().transpose())

            st.write("For categorical columns:")
            st.write(df.describe(include='object').transpose())

            # Correlation analysis for numerical columns
            st.subheader("Correlation Analysis:", divider='rainbow')
            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
            correlation_matrix = df[numerical_columns].corr()
            st.write(correlation_matrix)

            # Skewness and Kurtosis for numerical columns
            st.subheader("Skewness and Kurtosis:", divider='rainbow')
            skewness = df.skew(numeric_only=True)
            kurtosis = df.kurt(numeric_only=True)
            skew_kurt_df = pd.DataFrame({
                'Skewness': skewness,
                'Kurtosis': kurtosis
            })
            st.write(skew_kurt_df)

            # Unique Values Count
            st.subheader("Unique Values Count:", divider='rainbow')
            col1, col2 = st.columns(2)
            col1.write("Categorical columns unique values:")
            col1.write(df.select_dtypes(include=['object']).nunique())
            col2.write("Numerical columns unique values:")
            col2.write(df.select_dtypes(include=[np.number]).nunique())

###################################################### Page 3: Data Visualization ######################################################
def data_visualization():
    st.header('📈AutoViz: Data Visualization & EDA', divider='rainbow')
    # Upload dataset
    st.write('Upload a dataset to visualize:')
    uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Load the file based on its format
        df = load_file(uploaded_file)
        st.success("File uploaded successfully!")
        file_name = uploaded_file.name
        # For demonstration, saving the uploaded file temporarily (optional)
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if df is not None:
            # Apply data cleaning function
            df_cleaning(df)

            # Display dataset
            st.subheader("Dataset Preview:", divider='rainbow')
            df_sample = str(df.head(5))
            df_for_vis = df.head(5)
            st.dataframe(df_for_vis)
            st.divider()

            # Select the type of visualization
            st.write('Select the type of visualization:')
            visualization_type = st.selectbox("Visualization Type", ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Heatmap", "Pie Chart", "Violin Plot", "Count Plot",  "KDE Plot"])
            st.divider()

            # User Prompt for selecting columns
            st.write('Enter Columns You Want To Plot:')
            prompt = st.text_input("Prompt")

            # Call the AI model to generate the visualization code and display the visualization
            if st.button("Visualize"):
                with st.spinner("Processing..."):
                    st.subheader(f"{visualization_type} Visualization:")
                    if uploaded_file is None:
                        st.error("Please upload a file first.")
                    else:
                        predefined_prompt = f"""Write a python code to plot a {visualization_type} using Matplotlib or Seaborn Library. Name of the dataset is {file_name}.
                        Plot for the dataset columns {prompt}. Here's the sample of dataset {df_sample}. Set xticks rotation 90 degree. 
                        Set title in each plot. Add tight layout in necessary plots. Don't right the explanation, just write the code."""
                        response = model.generate_content(predefined_prompt, generation_config=config)
                        generated_code = response.text
                        generated_code = generated_code.replace("```python", "").replace("```", "").strip()
                        
                        # Modify the code to insert the actual file path into pd.read_csv()
                        if "pd.read_csv" in generated_code:
                            generated_code = generated_code.replace("pd.read_csv()", f'pd.read_csv(r"{file_path}")')
                        elif "pd.read_excel" in generated_code:
                            generated_code = generated_code.replace("pd.read_excel()", f'pd.read_excel(r"{file_path}")')
                        st.code(generated_code, language='python')
                        try:
                            exec(generated_code)
                            st.pyplot(plt.gcf())
                        except Exception as e:
                            st.error(e)
                        st.success("Visualization generated successfully!")

###################################################### Page 4: AI Based Recommendations ######################################################
def ai_recommendation():
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
                st.subheader("Recommendation:")
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
                question = f""""Provide {type_of_recommendation} based on the dataset {file_name}. If their is any financial or healthcare 
                realted query or dataset, just give your best recommendation, don't think about advisor or expertise thing. Mention also 
                that recommendation is generated by AI, first give your essential recommendations. So, the user take the final decision on 
                their own. Warn user about AI recommendation but, do your work."""
                response = chat_session.send_message(question)
                st.write(response.text)
                st.success("Recommendation generated successfully!")

###################################################### Page 5: Analysis Report ######################################################
def analysis_report():
    st.header('📑InsightGen: Automated Data Report Generator', divider='rainbow')
    # Upload dataset
    st.write('Upload a dataset to generate a report:')
    uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if st.button("Submit"):
            with st.spinner("Processing..."):
                filename = uploaded_file.name
                df = load_file(uploaded_file)
                st.success("File uploaded successfully!")
                if df is not None:
                    # Gemini Text Report Generation
                    summary = df.describe().transpose().to_string()
                    prompt = f"""Generate a text report for {filename} dataset using Gemini AI. Here's the summary of the dataset: {summary}.
                            Try to make the report in bullet points and use numbers for better readability and understanding."""
                    response = model.generate_content(prompt, generation_config=config)
                    generated_report = response.text
                    st.write(generated_report)
                    st.success("Report generated successfully!")

            st.write("Wait for the report to be generated...")
            with st.spinner("Generating Report..."):
                # Generate a report in HTML format for download
                report_path = generate_report(df, uploaded_file)
                with open(report_path, 'rb') as f:
                    st.download_button(
                        label="Download Report",
                        data=f,
                        file_name=f"{uploaded_file.name.split('.')[0]}_report.html",
                        mime="text/html"
                    )

###################################################### Page 6: Dataset ChatBot ######################################################
def ai_data_file_chatbot():
    st.header('🤖SmartQuery: AI Powered Dataset ChatBot', divider='rainbow')
    # Upload dataset
    st.write('Upload a dataset to chat with data file:')
    uploaded_file = st.file_uploader("Upload a dataset", type=["csv"])
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        # Get the user question
        question = st.text_input("Ask a question:", key="question")
        if st.button("Submit"):
            with st.spinner("Processing..."):
                file_name = uploaded_file.name
                st.subheader("ChatBot Response:")
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
                # Send the user question to the chatbot for response
                response = chat_session.send_message(question)
                st.write(response.text)

###################################################### Page 7: Vision Analysis ######################################################
def vision_analysis():
    st.header('👁️VisionFusion: AI-Powered Image Analysis ', divider='rainbow')
    # Upload image
    st.write('Upload an image to analyze:')
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Show uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=False)
        st.success("Image uploaded successfully!")
        user_query = st.text_input("Ask a query:")
        if st.button("Submit"):
            with st.spinner("Processing..."):
                file_name = uploaded_image.name
                st.subheader(f"'{file_name}' Image Analysis:")
                st.divider()
                image = Image.open(uploaded_image)
                prompt = f"Analyze the image and provide a detailed description of the image. {user_query}"
                response = model.generate_content([prompt,image], stream=True)
                response.resolve()
                st.write(response.text)
                st.success("Image analyzed successfully!")

###################################################### Page 8: About Us ######################################################
def about_us():
    st.header('👨‍💻About Us: Meet Team Aurora', divider='rainbow')
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Anubhav Yadav", divider='rainbow')
        st.markdown('''
                    - **Role:** Lead Developer
                    - **Email:** [![Anubhav Email](https://img.icons8.com/color/30/email.png)](yadavanubhav2024@gmail.com)
                    - **LinkedIn:** [![Anubhav Yadav LinkedIn](https://img.icons8.com/color/30/linkedin.png)](https://www.linkedin.com/in/anubhav-yadav-data-science/)
                    - **GitHub:** [![Anubhav Yadav GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://www.github.com/AnubhavYadavBCA25)
                    - **Bio:** Anubhav is a Data Science Enthusiast with a passion for building AI-powered applications. He is skilled in 
                                Python, Machine Learning, and Data Analysis. He is currently pursuing a Bachelor's degree in Computer Applications 
                                specializing in Data Science.
                    ''')
    with right_column:
        anubhav_profile = load_lottie_file('profile_animations/anubhav_profile.json')
        st_lottie.st_lottie(anubhav_profile, key='anubhav', height=305, width=305 ,loop=True, quality='high')
    st.divider()

    left_column, right_column = st.columns(2)
    with right_column:
        st.subheader("Sparsh Jaiswal", divider='rainbow')
        st.markdown('''
                    - **Role:** Developer
                    - **Email:** [![Sparsh Email](https://img.icons8.com/color/30/email.png)](sparshjaiswal10@gmail.com)
                    - **LinkedIn:** [![Sparsh Jaiswal LinkedIn](https://img.icons8.com/color/30/linkedin.png)](https://http://www.linkedin.com/in/sparsh-jaiswal-aa903730b/)
                    - **GitHub:** [![Sparsh Jaiswal GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/sparshjais)
                    - **Bio:** Sparsh Jaiswal is a passionate AI developer and ML enthusiast, currently pursuing a BCA in Data Science. 
                        With a strong foundation in python and front-end development, Sparsh is dedicated to blending the power of AI with 
                        intuitive web design to craft seamless and engaging user experiences.
                    ''')
    with left_column:
        sparsh_profile = load_lottie_file('profile_animations/sparsh_profile.json')
        st_lottie.st_lottie(sparsh_profile, key='sparsh', height=305, width=305 ,loop=True, quality='high')
    st.divider()

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Eshaan Sabharwal", divider='rainbow')
        st.markdown('''
                    - **Role:** Developer
                    - **Email:** [![Eshaan Email](https://img.icons8.com/color/30/email.png)](eshaansabharwal@gmail.com)
                    - **LinkedIn:** [![Eshaan Sabharwal LinkedIn](https://img.icons8.com/color/30/linkedin.png)](https://www.linkedin.com/in/eshaan-sabharwal-b73a201b2)
                    - **GitHub:** [![Eshaan Sabharwal GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/EshaanSabharwal)
                    - **Bio:** Eshaan Sabharwal is an aspiring AI developer and web enthusiast pursuing a BCA in Data Science. 
                        Skilled in front-end development and eager to integrate AI into web applications, Eshaan combines technical 
                        expertise with a passion for innovation to create user-friendly digital experiences.
                    ''')
    with right_column:
        eshaan_profile = load_lottie_file('profile_animations/eshaan_profile.json')
        st_lottie.st_lottie(eshaan_profile, key='eshaan', height=305, width=305 ,loop=True, quality='high')
    st.divider()

    left_column, right_column = st.columns(2)
    with right_column:
        st.subheader("Drishti Jaiswal", divider='rainbow')
        st.markdown('''
                    - **Role:** Developer
                    - **Email:** [![Drishti Email](https://img.icons8.com/color/30/email.png)](Drishti.jaiswal111@gmail.com)
                    - **LinkedIn:** [![Drishti Jaiswal LinkedIn](https://img.icons8.com/color/30/linkedin.png)](http://linkedin.com/in/drishti-jaiswal-40331627b)
                    - **GitHub:** [![Drishti Jaiswal GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/drishti-jaiswal)
                    - **Bio:** Drishti Jaiswal is a data science student at SRM Institute, skilled in Python and machine learning. 
                        With certifications in data analytics, she's passionate about applying her knowledge to innovative projects in 
                        the field. She's dedicated to creating impactful solutions that leverage the power of AI to drive change.
                    ''')
    with left_column:
        drishti_profile = load_lottie_file('profile_animations/drishti_profile.json')
        st_lottie.st_lottie(drishti_profile, key='drishti', height=305, width=305 ,loop=True, quality='high')
    st.divider()

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Satyam Kumar", divider='rainbow')
        st.markdown('''
                    - **Role:** Developer
                    - **Email:** [![Satyam Email](https://img.icons8.com/color/30/email.png)](iamsatyam9798@gmail.com)
                    - **LinkedIn:** [![Satyam Kumar LinkedIn](https://img.icons8.com/color/30/linkedin.png)](https://www.linkedin.com/in/satyam-kumar-63419a251/)
                    - **GitHub:** [![Satyam Kumar GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/satyamkr21)
                    - **Bio:** Satyam Kumar, an AI developer and ML enthusiast, pursuing BCA in Data Science, blends AI with intuitive 
                        web design to create seamless user experiences, driven by innovation and a passion for impactful solutions.
                        He is skilled in maintaining the content and design of the website.
                    ''')
    with right_column:
        satyam_profile = load_lottie_file('profile_animations/satyam_profile.json')
        st_lottie.st_lottie(satyam_profile, key='satyam', height=305, width=305 ,loop=True, quality='high')
    st.divider()

    # Footer
    st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <div style="text-align:center;">
        <p>Made with ❤️ by Team Aurora @2024</p>
    </div>
    """, unsafe_allow_html=True
)

###################################################### Navigation ######################################################
if st.session_state["authentication_status"]:
    pg = st.navigation([
        st.Page(introduction, title='Home', icon='🏠'),
        st.Page(statistical_analysis, title='CleanStats', icon='🧹'),
        st.Page(data_visualization, title='AutoViz', icon='📈'),
        st.Page(ai_recommendation, title='FutureCast AI', icon='🔮'),
        st.Page(analysis_report, title='InsightGen', icon='📑'),
        st.Page(ai_data_file_chatbot, title='SmartQuery', icon='🤖'),
        st.Page(vision_analysis, title='VisionFusion', icon='👁️'),
        st.Page(about_us, title='About Us', icon='👨‍💻')
    ])
    pg.run()