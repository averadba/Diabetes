# Required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pycaret.datasets import get_data
from pycaret.classification import load_model, predict_model
import streamlit as st

# Defining Prediction Function

def predict_rating(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    
    return predictions_data['Label'][0]




# Loading Model

model = load_model('Final_Model_xtra_tree')


# Writing the title of the app a a brief description.

st.write("""
# Diabetes Prediction App

Use this app to predict if the female patient will develop diabetes.


""")

# Making Sliders and Feature Variables

times_pregnant = st.sidebar.slider(label = 'Times Pregnant', min_value=0, max_value=20,
                                    value=0, step=1)

glucose = st.sidebar.slider(label = 'Plasma Glucose Concentration', min_value=0,
                            max_value=200, value=120, step=1)

diastolic = st.sidebar.slider(label = 'Diastolic Blood Pressure', min_value=0, max_value=230,
                                    value=69, step=1)

thickness = st.sidebar.slider(label = 'Triceps Skinfold Thickness', min_value=0, max_value=120,
                                    value=20, step=1)

insulin = st.sidebar.slider(label = '2-Hour Serum Insulin', min_value=0, max_value=2000,
                                    value=80, step=1)

bmi = st.sidebar.slider(label = 'Body Mass Index', min_value=0, max_value=150,
                                    value=30, step=1)

pedigree = st.sidebar.slider(label = 'Diabetes Pedigree Function', min_value=0.00, max_value=10.00,
                                    value=0.50, step=0.01)

age = st.sidebar.slider(label = 'Age in Years', min_value=15, max_value=90,
                                    value=21, step=1)


# Mapping feature labels with slider values

features = {'times_pregnant':times_pregnant,
            'glucose':glucose,
            'diastolic':diastolic,
            'thickness':thickness,
            'insulin':insulin,
            'bmi':bmi,
            'pedigree':pedigree,
            'age':age
}


# Converting Features into DataFrame

features_df  = pd.DataFrame([features])

st.table(features_df)


# Predicting Diabetes Risk

if st.button('Predict'):
    
    prediction = predict_rating(model, features_df)
    
    st.write(' Based on input data, the diabetes risk is '+ prediction)