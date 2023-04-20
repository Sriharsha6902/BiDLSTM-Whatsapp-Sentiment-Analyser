#!/usr/bin/env python
# coding: utf-8

# In[25]:


import re
import pandas as pd
import keras
best_model = keras.models.load_model("Sentiment_analysis_BiLSTM.hdf5")


# In[26]:


f = open('WhatsApp Chat with Avinash Vnr.txt','r',encoding='utf-8')


# In[27]:


data = f.read()


# In[28]:


pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'


# In[29]:


messages = re.split(pattern,data)[1:]
len(messages)


# In[30]:


dates = re.findall(pattern,data)
len(dates)


# In[31]:


df = pd.DataFrame({'user_message':messages, 'message_date': dates})
#convert message_date type
df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %H:%M - ')
df.rename(columns={'message_date': 'date'}, inplace=True)
df.head()


# In[32]:


df['year'] = df['date'].dt.year


# In[33]:


df['month'] = df['date'].dt.month_name()


# In[34]:


df['day'] = df['date'].dt.day


# In[35]:


df['hour'] = df['date'].dt.hour


# In[36]:


df['minute'] = df['date'].dt.minute


# In[37]:


df.head()


# In[38]:


df=df[1:]


# In[39]:


df


# In[40]:


df1=df


# In[41]:


# iterate over each sentence in the dataframe column
for sentence in df['user_message']:
    # split the sentence into username and message using ':'
    split_sentence = sentence.split(':')
    if len(split_sentence)==1:
        r=df.index[df['user_message']==sentence].tolist()[0]
        df=df.drop(index=r)


# In[42]:


df


# In[43]:


usernames = []
messages = []
for sentence in df['user_message']:
    split_sentence = sentence.split(':')
    usernames.append(split_sentence[0])
    if split_sentence[1]:
        messages.append(split_sentence[1])
df['username'] = usernames
df['message'] = messages
df=df.drop('user_message',axis=1)


# In[44]:


for sen in df['message']:
    if sen==" <Media omitted>\n" or sen==" You deleted this message\n" or sen==" This message was deleted\n":
        r=df.index[df['message']==sen].tolist()[0]
        df=df.drop(index=r)
df


# In[45]:


df2=df['message']


# In[46]:


df2.to_csv('Messages.csv', encoding='utf-8', index=False)

