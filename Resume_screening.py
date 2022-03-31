#!/usr/bin/env python
# coding: utf-8
# Import required libraries
import PyPDF2
import textract
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import docx2txt
import streamlit as st
import time
import os
import pdfplumber
import plotly.express as px

#selenium related library
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from parsel import Selector 

#nlp library for extracting name from CV
import spacy
from spacy.matcher import Matcher
from io import StringIO
from collections import Counter
from spacy.matcher import PhraseMatcher

#fix an error
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# load pre-trained model
nlp = spacy.load('en_core_web_sm')

# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)

# Create dictionary with industrial and system engineering key terms by area
terms = {'Quality/Six Sigma':['black belt','capability analysis','control charts','doe','dmaic','fishbone',
                              'gage r&r', 'green belt','ishikawa','iso','kaizen','kpi','lean','metrics',
                              'pdsa','performance improvement','process improvement','quality',
                              'quality circles','quality tools','root cause','six sigma',
                              'stability analysis','statistical analysis','tqm'],      
        'Operations management':['automation','bottleneck','constraints','cycle time','efficiency','fmea',
                                 'machinery','maintenance','manufacture','line balancing','oee','operations',
                                 'operations research','optimization','overall equipment effectiveness',
                                 'pfmea','process','process mapping','production','resources','safety',
                                 'stoppage','value stream mapping','utilization'],
        'Supply chain':['abc analysis','apics','customer','customs','delivery','distribution','eoq','epq',
                        'fleet','forecast','inventory','logistic','materials','outsourcing','procurement',
                        'reorder point','rout','safety stock','scheduling','shipping','stock','suppliers',
                        'third party logistics','transport','transportation','traffic','supply chain',
                        'vendor','warehouse','wip','work in progress'],
        'Project management':['administration','agile','budget','cost','direction','feasibility analysis',
                              'finance','kanban','leader','leadership','management','milestones','planning',
                              'pmi','pmp','problem','project','risk','schedule','scrum','stakeholders'],
        'Data analytics':['analytics','api','aws','big data','busines intelligence','clustering','code',
                          'coding','data','database','data mining','data science','deep learning','hadoop',
                          'hypothesis test','iot','internet','machine learning','modeling','nosql','nlp',
                          'predictive','programming','python','r','sql','tableau','text mining',
                          'visualuzation']}

#create chrome driver option and path:
PATH = "chromedriver.exe" 

options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--disable-features=NetworkService")
options.add_argument("--window-size=1920x1080")
options.add_argument("--disable-features=VizDisplayCompositor")

def save_uploadedfile(uploadedfile):
    dataType = uploadedfile.type
    if(dataType == 'image/png' or 'image/jpeg' or 'image/jpeg'):
        with open(os.path.join("./", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())

    if(dataType == 'text/csv'):
        with open(os.path.join("./", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())

    if(dataType == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or 'text/plain' or 'application/pdf'):
        with open(os.path.join("./", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())

    else:
        with open(os.path.join("tempDir", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())

    return st.success("Saved File: {} to tempDir".format(uploadedfile.name))

def text_cleansing(text):

    # Convert all strings to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+','',text)

    # Remove punctuation
    text = text.translate(str.maketrans('','',string.punctuation))

    return text

def convert_txt_to_df(text):
    
    # Convert all strings to lowercase
    text_ = text.lower()

    # Remove numbers
    text_ = re.sub(r'\d+','',text_)

    # Remove punctuation
    text_ = text_.translate(str.maketrans('','',string.punctuation))
    
    # Initializie score counters for each area
    quality = 0
    operations = 0
    supplychain = 0
    project = 0
    data = 0
    
    # Create an empty list where the scores will be stored
    scores = []

    # Obtain the scores for each area
    for area in terms.keys():
        
        if area == 'Quality/Six Sigma':
            for word in terms[area]:
                if word in text:
                    quality +=1
            scores.append(quality)
        
        elif area == 'Operations management':
            for word in terms[area]:
                if word in text:
                    operations +=1
            scores.append(operations)
        
        elif area == 'Supply chain':
            for word in terms[area]:
                if word in text:
                    supplychain +=1
            scores.append(supplychain)
        
        elif area == 'Project management':
            for word in terms[area]:
                if word in text:
                    project +=1
            scores.append(project)
        
        elif area == 'Data analytics':
            for word in terms[area]:
                if word in text:
                    data +=1
            scores.append(data)
            
    # Create a data frame with the scores summary
    summary = pd.DataFrame(scores,index=terms.keys(),columns=['score']).sort_values(by='score',ascending=False)
    
    return summary

def plot_df(df):

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    explode = (0.1, 0.1, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(df['score'], explode=explode, labels=df.index, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)

def upload_validate(file):
    
    if file is not None:
        file_details = {"Filename": file.name,
                        "FileType": file.type, "FileSize": file.size}
        #st.write(file_details)

        with st.spinner('Uploading Files...'):
        
            #plain text
            if file.type == "text/plain":
            
                st.text(str(file.read(), "utf-8"))
                raw_text = str(file.read(), "utf-8")
                time.sleep(1)
                #save_uploadedfile(file)
                return raw_text

            #pdf
            elif file.type == "application/pdf":
                try:
                    with pdfplumber.open(file) as pdf:
                        page = pdf.pages[0]
                        time.sleep(1)
                        #save_uploadedfile(file)
                        return page.extract_text()

                except:
                    st.write("None")
                    #docx
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                
                raw_text = docx2txt.process(file)
                time.sleep(1)
                return raw_text

def login_2_linkedin(driver):
    
    driver.get("https://www.linkedin.com")
    sleep(1)

    username = driver.find_element_by_class_name('input__input')
    username.send_keys('edward_lam@vfc.com')
    sleep(0.5)

    password = driver.find_element_by_id('session_password')
    password.send_keys('Today2022') 
    sleep(0.5)

    log_in_button = driver.find_element_by_class_name('sign-in-form__submit-button') 
    log_in_button.click()

def search_linkedin_url(driver, name):
    
    driver.get("https://www.google.com")
    sleep(3)

    #search_query = driver.find_element_by_name('q')
    search_query = driver.find_element(by=By.NAME, value='q')
    search_query.send_keys('site:linkedin.com/in/ AND "' + name + '"')
    sleep(0.5)

    search_query.send_keys(Keys.RETURN)
    sleep(1)

    #linkedin_urls = driver.find_elements_by_class_name('yuRUbf')
    myLinks=driver.find_elements_by_xpath('//*[@id="rso"]/div[1]/div/div/div[1]/div/a')

    #candidate_urls = [url.get_attribute('href') for url in linkedin_urls]
    sleep(0.5)

    candidate_url = []

    for link in myLinks:
        candidate_url.append(link.get_attribute("href"))
    
    return candidate_url

def extract_name_from_cv(resume_text):

    nlp_text = nlp(resume_text)
    
    # First name and Last name are always Proper Nouns
    # First name and Last name are always Proper Nouns
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    pattern3 = [{'POS': 'PROPN'}, {"OP": "?"},  {"OP": "?"}, {'POS': 'PROPN'}]

    matcher.add('NAME', [pattern])
    matcher.add('Name variation 1', [pattern3])
    
    d = []
    matches = matcher(nlp_text)
    
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
        span = nlp_text[start : end]  # get the matched slice of the doc
        #print(span.text)
        if end<10:
            d.append((rule_id, span.text, end - start, end))      

    keywords = "\n".join(f'{i[0]} / {i[1] } / {i[2] } / {i[3] }' for i,j in Counter(d).items())
    
    df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])
    df1 = pd.DataFrame(df.Keywords_List.str.split('/',4).tolist(),columns = ['Subject','Keyword', 'length', 'end'])
    dfObj = df1.sort_values(by ='length', ascending=False)
    
    print(dfObj)

    return dfObj['Keyword'].iloc[0]
   
def getLinkedinInterest(driver, url):

    interest_url = url + '/details/interests/'
    st.write(interest_url)
    
    interest, interest_desc = get_content_from_url(interest_url, driver)
    interest_list = breakdown_list(interest, interest_desc)

    return interest_list

from itertools import zip_longest
#build helper function
def grouper(iterable, n, fillvalue=None):
# Collect data into fixed-length chunks or blocks"
# grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def get_content_from_url(exp_url, driver):
    
    full_list = []
    item_list = []

    driver.get(exp_url)
    sleep(3)

    sel = Selector(text=driver.page_source)
    
    for span in sel.xpath("//span[contains(@class, 'visually-hidden')]"):
    
        full_list.append(span.xpath('./text()').get())

    for span in sel.xpath("//span[contains(@class, 't-bold mr1')]/span[contains(@class, 'visually-hidden')]"):
    
        item_list.append(span.xpath('./text()').get())
        
    return full_list, item_list

def breakdown_list(full_list, item_list):
    
    final_list = []
    sub_list = []
    
    for item in full_list[1:]:
        if item != '\n      Status is ':
            if any(item == s for s in item_list):
            
                final_list.append(sub_list)
                sub_list = []
                
            sub_list.append(item)

    return final_list[1:]

def jdlist_cleansing(jd_dict):
    new_dict = {}

    for k, v in jd_dict.items():
        
        newlist = []
        for x in jd_dict[k]: 
            if str(x) != 'nan':
                x = nlp(x)
                newlist.append(x)
    
        new_dict[k] = newlist
    
    return new_dict

def produce_ability_matrix(jd, cv):

    matcher = PhraseMatcher(nlp.vocab)

    for k, v in jd.items():
    
        matcher.add(k, None, *v)

    text = str(cv)
    text = text.replace("\\n", "")
    text = text.lower()
    doc = nlp(text)

    d = []
    matches = matcher(doc)
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
        span = doc[start : end]  # get the matched slice of the doc
        #print(span.text)
        d.append((rule_id, span.text))      

    keywords = "\n".join(f'{i[0]} / {i[1]} ({j})' for i,j in Counter(d).items())

    ## convertimg string of keywords to dataframe
    df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])
    df1 = pd.DataFrame(df.Keywords_List.str.split('/',1).tolist(),columns = ['Subject','Keyword'])
    df2 = pd.DataFrame(df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])
    df3 = pd.concat([df1['Subject'],df2['Keyword'], df2['Count']], axis =1) 
    df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))

    return df3

def main():
    #start the flow

    st.title("Profile Screening")

    st.write("This app takes resume in word document, perform analysis on candidates' background using resume & his Linkedin profile")

    with st.sidebar:

        jd_file = st.file_uploader(
            "Choose JD Doc", type=['xlsx', 'csv'], key="document_file_uploader")
                
        docx_file = st.file_uploader(
            "Choose Resume Doc", type=['txt', 'docx', 'pdf'], key="document_file_uploader")

    #Start conversion
    but1, but2, but3= st.columns(3)

    if docx_file is not None:

        # Create a simple button
        if but1.button('Skill set distribution'):
            raw_text = upload_validate(docx_file)
            text = text_cleansing(raw_text)
            data_frame = convert_txt_to_df(text)
            plot_df(data_frame)

        if(but2.button('Linkedin Interest')):

            driver = webdriver.Chrome(PATH, options=options) 
            raw_text = upload_validate(docx_file)
            text = text_cleansing(raw_text)
            print(text)
            name = extract_name_from_cv(text.replace('\n', ' ').replace('\t', ' '))
        
            if name != '':
                st.write("Searching linkedin page using name : " + name + "...")
                url = search_linkedin_url(driver, name)
                st.write(url[0])
                login_2_linkedin(driver)
                st.write(getLinkedinInterest(driver, url[0]))

            else: 
                st.write('we cannot find the name from the CV')

        if(but3.button('JD Matching')):
            #read in the cv
            raw_text = upload_validate(docx_file)
            text = text_cleansing(raw_text)
            
            #read in the jd
            if jd_file is not None:
                jd = pd.read_excel(jd_file)
                jd_dict = jd.to_dict('list')
                new_dict = jdlist_cleansing(jd_dict)
                df = produce_ability_matrix(new_dict, text)
                st.write(df)
            else:
                st.text('Please select a valid jd in the left panel...')


    else:
        st.text('Please select a valid summary in the left panel...')


if __name__=='__main__':
    main()    