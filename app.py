import joblib 
import streamlit as st
import numpy as np

# import all necessary models
fc_model = joblib.load('FutureCareer_en.pkl')
gend_model = joblib.load('gender_en.pkl')
id_model = joblib.load('InterestedDomain_en.pkl')
lang_model = joblib.load('lang_en.pkl')
pred_en = joblib.load('output_model.pkl')
Projects_en = joblib.load('Projects_en.pkl')


# creating a simple UI
st.header('Computer Science Student Carrier Recommender')
st.text('')

age = st.text_input('Enter your age : ')
if age:
    try:
        age = int(age)
    except ValueError:
        st.error(' Please enter a valid age')

gender = st.radio('Enter your gender', ['Male','Female'])

gpa = st.text_input('Enter your CGPA : ')
if gpa:
    try:
        gpa = float(gpa)
    except ValueError:
        st.error('Please enter a valid CGPA')

id = st.selectbox('Select your interested domain', ['Artificial Intelligence', 'Data Science', 'Software Development',
       'Web Development', 'Cybersecurity', 'Machine Learning',
       'Database Management', 'Cloud Computing', 'Mobile App Development',
       'Computer Graphics', 'Software Engineering', 'Network Security',
       'Game Development', 'Computer Vision', 'Bioinformatics',
       'IoT (Internet of Things)', 'Natural Language Processing',
       'Data Mining', 'Human-Computer Interaction',
       'Biomedical Computing', 'Quantum Computing',
       'Blockchain Technology', 'Information Retrieval', 'Data Privacy',
       'Geographic Information Systems', 'Distributed Systems',
       'Digital Forensics'])

project = st.text_input('Enter your one project')

python = st.selectbox('What is your python knowledge level', ['Weak', 'Average', 'Strong'])
sql = st.selectbox('What is your sql knowledge level', ['Weak', 'Average', 'Strong'])
java = st.selectbox('What is your java knowledge level', ['Weak', 'Average', 'Strong'])

st.text('')


# converting categorical data to lower case
gender = gender.lower()
id = id.lower()
project = project.lower()
python = python.lower()
sql = sql.lower()
java = java.lower()


# encoding categorical data into numerical data
gender = gend_model.transform([gender])[0]
project = Projects_en.transform([[project]])[0]
python = lang_model.transform([[python]])[0][0]
sql = lang_model.transform([[sql]])[0][0]
java = lang_model.transform([[java]])[0][0]
id = id_model.transform([id])[0]


# predicting output
features = np.array([gender, age, gpa, id, python, sql, java])
features = np.concatenate((features, project)).reshape(1, -1)

if st.button('recommend Carrier'):
    y = pred_en.predict(features)
    y = fc_model.inverse_transform(y)
    st.text('')
    st.success(f'You can become a {y[0]}')