#!/usr/bin/env python
# coding: utf-8

# In[5]:


import joblib
import streamlit as st
def predict(data):
    clf = joblib.load("rf_model.sav")
    return clf.predict(data)

#from prediction import predict
if st.button ("Predict"):  
    result = prediction(radius_mean, texture_mean, perimeter_mean)  
    st.success ('The output of the above is {}'.format(result))
    


# In[ ]:





# In[ ]:




