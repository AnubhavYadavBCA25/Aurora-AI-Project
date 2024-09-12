import pickle
from pathlib import Path
import streamlit_authenticator as stauth

names = ['Anubhav', 'Eshaan', 'Drishti']
usernames = ['anubhav@2024', 'eshaan@2004', 'drishti@2022']
passwords = ['00', '00', '00']

hased_passwords = stauth.Hasher(passwords).generate()

print(hased_passwords)