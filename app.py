#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import pickle
import streamlit as smt
from PIL import Image
import sklearn  
# loading in the model to predict on the data
pickle_in = open('model.pkl', 'rb')
model = pickle.load(pickle_in)
#model=pd.read_csv("C:\Users\BRAVE BARAKA\Breast cancer prediction\data.csv")
def welcome():
    return 'welcome all'
  
# defining the function which will make the prediction using 
# the data which the user inputs
def prediction(radius_mean, texture_mean, perimeter_mean):  
   
    prediction=model.predict([[radius_mean,texture_mean,perimeter_mean]])
    print(prediction)
    return prediction
      
  
# this is the main function in which we define our webpage 
def main():
      # giving the webpage a title
    smt.title("Breast Cancer Prediction")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Breast Cancer Classifier ML App </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    smt.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    radius_mean = smt.number_input ("radius_mean") 
    texture_mean = smt.number_input ("texture_mean")
    perimeter_mean = smt.number_input ("perimeter_mean")
    X = pd.DataFrame([[radius_mean, texture_mean, perimeter_mean]], 
                     columns = ["radius_mean", "texture_mean", "perimeter_mean"])
    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if smt.button("Predict"):
        prediction = model.predict(X)[0]
        #smt.success('The output is {}'.format(result))
     # Output prediction
        #X = X.replace(["M", "B"], [1, 0])
        smt.text(f"This instance is a {prediction}")
        print(X)
     
if __name__=='__main__':
    main()


# In[ ]:




