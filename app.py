import yaml
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import LoginError
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Streamlit page configuration
st.set_page_config(
    page_title="Aurora AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

#-----------------------------User Authentication-----------------------------#

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

#-------------------------------- AI Models-----------------------------------#
# Gemini API
load_dotenv()
genai_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=genai_api_key)
model = genai.GenerativeModel('gemini-1.5-flash')
config = genai.types.GenerationConfig(temperature=1.0, max_output_tokens=300)

#----------------------------- Data Loading & Cleaning Functions -----------------------------#
# Function for loading csv format file:
def load_csv_format(file):
        df = pd.read_csv(file)
        return df
    
# Function for loading xlsx format file:
def load_xlsx_format(file):
        df = pd.read_excel(file)
        return df

# Function for loading file based on its format:
def load_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
            return load_csv_format(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
            return load_xlsx_format(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or XLSX file.")
    st.success("File uploaded successfully!")

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

#----------------------------- Introduction Page -----------------------------#
def introduction():
    st.header('🤖Aurora: AI Powered Data Analysis Tool', divider='rainbow')
    st.subheader("Introduction")
    st.write("Welcome to Aurora AI, an AI powered data analysis tool.")

#----------------------------- Page 1: Statistical Analysis -----------------------------#
def statistical_analysis():
    st.header('🧹CleanStats: Cleaning & Statistical Analysis', divider='rainbow')
    
    # Upload dataset
    uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx"])

    if uploaded_file is not None:
    # Load the file based on its format
        df = load_file(uploaded_file)

        if df is not None:
        # Remove duplicate rows
            df_cleaning(df)

            # Display dataset
            st.subheader("Dataset Preview:", divider='rainbow')
            st.dataframe(df)
        
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

#----------------------------- Page 2: Data Visualization -----------------------------#
def data_visualization():
    st.header('📈AutoViz: Data Visualization & EDA', divider='rainbow')
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
            st.dataframe(df)
            df_sample = str(df.head(5))
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
                if uploaded_file is None:
                    st.error("Please upload a file first.")
                else:
                    predefined_prompt = f"""Write a python code to plot a {visualization_type} using Matplotlib or Seaborn Library. Name of the dataset is {file_name}.
                    Plot for the dataset columns {prompt}. Here's the sample of dataset {df_sample}. Set xticks rotation 90 degree. 
                    Set title in each plot. Don't right the explanation, just write the code."""
                    response = model.generate_content(predefined_prompt, generation_config=config)
                    generated_code = response.text
                    generated_code = generated_code.replace("```python", "").replace("```", "").strip()
                    # Step 5: Modify the code to insert the actual file path into pd.read_csv()
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

#----------------------------- Page 3: Predictive Analysis -----------------------------#
def predictive_analysis():
    st.header('Predictive Analysis', divider='rainbow')

#----------------------------- Page 4: Analysis Report -----------------------------#
def analysis_report():
    st.header('Analysis Report', divider='rainbow')

#----------------------------- Page 5: AI Recommendations -----------------------------#
def ai_recommendations():
    st.header('AI Recommendations', divider='rainbow')

#----------------------------- Navigation -----------------------------#
if st.session_state["authentication_status"]:
    pg = st.navigation([
        st.Page(introduction, title='Home', icon='🏠'),
        st.Page(statistical_analysis, title='CleanStats', icon='🧹'),
        st.Page(data_visualization, title='AutoViz', icon='📈'),
        st.Page(predictive_analysis, title='Predictive Analysis', icon='🔮'),
        st.Page(analysis_report, title='Analysis Report', icon='📑'),
        st.Page(ai_recommendations, title='AI Recommendations', icon='🤖')
    ])
    pg.run()