#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:29:37 2024

@author: davidhernandez
"""

# Import libraries

import streamlit as st
import pandas as pd
import joblib

# Load our model pipeline object

model = joblib.load("model.joblib")


# Add title and instructions

st.title("Purchase Prediction Model")
st.subheader("Enter Customer Information and Submit for Likelihood to Purchase")

# Age input form

age = st.number_input(
    label = "01. Enter the customer's age",
    min_value = 18,
    max_value = 120,
    value = 35,   
    )

#value is the default or pre-populated value 

# Gender input form

gender = st.radio(
    label = "02. Enter the customer's gender",
    options = ['M','F'])

# Credit Score input form

credit_score = st.number_input(
    label = "03. Enter the customer's credit score",
    min_value = 0,
    max_value = 1000,
    value = 500,
    )

# Submit inputs to model

if st.button("Submitt for prediction"):
    
    # store our data in a dataframe for predictions
    new_data = pd.DataFrame({"age" : [age], "gender" : [gender], "credit_score" : [credit_score]})
    
    # apply model pipeline to input data and extract probability prediction
    pred_proba = model.predict_proba(new_data)[0][1]# Extract just numeric output from pred_proba
    
    # output prediction
    st.subheader(f"Based on these customer's attributes, our model predicts a purchase probability of {pred_proba:.0%}")
    












 

