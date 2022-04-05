from linkedin_api import Linkedin
import streamlit as st

# Authenticate using any Linkedin account credentials
api = Linkedin('edward_lam@vfc.com', 'Today2022')

# GET a profile
profile = api.get_profile('chong-fai-edward-lam-09285568')

# GET a profiles contact info
contact_info = api.get_profile_contact_info('chong-fai-edward-lam-09285568')

# GET 1st degree connections of a given profile
connections = api.get_profile_connections('chong-fai-edward-lam-09285568')

st.write(profile)