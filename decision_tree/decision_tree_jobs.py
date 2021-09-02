#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('./salaries.csv')


# In[3]:


df_target = df['salary_more_then_100k']


# In[4]:


df_2 = df.drop('salary_more_then_100k', axis=1)


# In[5]:


from sklearn.preprocessing import LabelEncoder


# In[6]:


encode = LabelEncoder()
df_2['company_1'] = encode.fit_transform(df_2['company'])
df_2['job_1'] = encode.fit_transform(df_2['job'])
df_2['degree_1'] = encode.fit_transform(df_2['degree'])


# In[7]:




# In[8]:


input_n = df_2.drop(['company', 'job','degree'], axis=1)


# In[14]:


from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(input_n, df_target, test_size=0.2, random_state=42)


# In[9]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[15]:


model.fit(x_train, y_train)


# In[16]:


model.score(x_test, y_test)


# In[20]:




# In[32]:


model.score(x_test, y_test)


# In[ ]:





# In[ ]:
## Streamlit web app

st.title('job app')

comp = st.selectbox(' Company',['Google', 'Pharma', 'Facebook'])
degree = st.selectbox(' Degree',['Bachlors', 'masters'])
job = st.selectbox(' Job',['sales executive', 'Business manager', 'computer programmer'])


def change(A,B,C):
    if (A == 'Google'):
        A = 2
    elif (A == 'Pharma'):
        A = 0
    elif (A == 'Facebook'):
        A = 1
    else:
        None
    if (B == 'Bachlors'):
        B = 0
    elif (B == 'masters'):
        B = 1
    else:
        None
    if (C == 'sales executive'):
        C = 2
    elif (C == 'Business manager'):
        C = 0
    elif (C == 'computer programmer'):
        C = 1
    else:
        None
    return A, B, C

    


#change('Facebook', 'masters', 'computer programmer' )
value = change(comp, degree, job )
#st.write(value)
m1 = value[0]
m2 = value[1]
m3 = value[2]

pre = model.predict([[m1,m3,m2]])
#st.write(pre)

def pri(N):

    if(N == 0):
        st.subheader('Salary is below 100k')
    else:
        st.subheader('Salary is above 100k')

st.write(' --------------------------------------------------')
pri(pre)
#st.write(m3)
