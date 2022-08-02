#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df=pd.read_csv("data.csv")
df.head(5)


# In[2]:


df.drop("id",axis=1,inplace=True)
df.drop("Unnamed: 32",axis=1,inplace=True)
df.columns


# In[3]:


y=df["diagnosis"]
x=df.iloc[:,1:]

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest,chi2
encoded=LabelEncoder()
y=encoded.fit_transform(y).astype(int)


#important features
chi2_features=SelectKBest(chi2,k=10)
x_reduced=chi2_features.fit_transform(x,y)
print(x_reduced.shape[1])
print(x.shape[1])


# In[4]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train,x_test,y_train,y_test=train_test_split(x_reduced,y,test_size=0.3,random_state=42)
RFC=RandomForestClassifier(n_estimators=5)
model=RFC.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy_score(y_test,y_pred)
for feature in zip(x.columns,model.feature_importances_):
    print(feature)


# In[5]:


accuracy_score(y_test,y_pred)


# In[7]:


import joblib
joblib.dump(model, "rf_model.sav")


# In[8]:


import pickle
pickle_out = open("model.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()


# In[10]:


print(df["diagnosis"])

