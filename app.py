import yaml
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import LoginError

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
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")

        # Remove duplicate rows
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

        # Display dataset
        st.subheader("Dataset Preview:", divider='rainbow')
        st.dataframe(df)
    
        # Basic statistics
        st.subheader("Basic Statistics", divider='rainbow')
        st.write("For numerical columns:")
        st.write(df.describe().transpose())

        st.write("For categorical columns:")
        st.write(df.describe(include='object').transpose())

        # Correlation analysis for numerical columns
        st.subheader("Correlation Analysis", divider='rainbow')
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        correlation_matrix = df[numerical_columns].corr()
        st.write(correlation_matrix)

        # Skewness and Kurtosis for numerical columns
        st.subheader("Skewness and Kurtosis", divider='rainbow')
        skewness = df.skew(numeric_only=True)
        kurtosis = df.kurt(numeric_only=True)
        skew_kurt_df = pd.DataFrame({
            'Skewness': skewness,
            'Kurtosis': kurtosis
        })
        st.write(skew_kurt_df)

        # Unique Values Count
        st.subheader("Unique Values Count:", divider='rainbow')
        st.write("Categorical columns unique values:")
        st.write(df.select_dtypes(include=['object']).nunique())
        st.write("Numerical columns unique values:")
        st.write(df.select_dtypes(include=[np.number]).nunique())

#----------------------------- Page 2: Data Visualization -----------------------------#
def data_visualization():
    st.header('Data Visualization', divider='rainbow')

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
        st.Page(statistical_analysis, title='Statistical Analysis', icon='📊'),
        st.Page(data_visualization, title='Data Visualization', icon='📈'),
        st.Page(predictive_analysis, title='Predictive Analysis', icon='🔮'),
        st.Page(analysis_report, title='Analysis Report', icon='📑'),
        st.Page(ai_recommendations, title='AI Recommendations', icon='🤖')
    ])
    pg.run()