
import tweepy as tw
import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

consumer_key = "Enter ur key code"
consumer_secret = "Enter ur key code"
access_token = "3172207346-Enter ur key code"
access_token_secret = "Enter ur key code"
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

st.title('Live Twitter Sentiment Analysis Brand Starbucks with Tweepy and VADER')
col1, col2 = st.columns([1,1])
with col1:
        st.image("logo STMI.png")

with col2:
		st.subheader('Kelompok Fine Report')
		st.markdown('Actabella')
		st.markdown('Nur Indriani Putri-1319073')
		st.markdown('Afa Maruti-1319109')
st.markdown('------')


# In[21]:


import matplotlib.pyplot as plt
import matplotlib
from plotly import graph_objs as go
import plotly.express as px

import seaborn as sns
import re
import os
import sys
import ast
plt.style.use('fivethirtyeight')
# Function for getting the sentiment
cp = sns.color_palette()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


st.subheader('Live Twitter Sentiment Analysis x Transformers')
query = "starbucks"
hasilSearch = api.search_tweets(q=query, count = 100, tweet_mode="extended", lang='en')

datat = [[tweet.user.screen_name, tweet.created_at, ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet.full_text).split())] for tweet in hasilSearch]

tweet_text = pd.DataFrame(data=datat, 
                    columns=['user', "tanggal", "isi"])
st.write('Hasil Live twitter asli')
tweet_text

df = tweet_text


# membersihkan data
def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
    text = re.sub('#', '', text) # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text) # Removing RT
    text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
    return text

df['isi'] = df['isi'].apply(cleanTxt)

#remove duplikat data
df.drop_duplicates(inplace = True)
st.write('Hasil Live twitter setelah cleaning data')
df

# Generating sentiment for all the sentence present in the dataset
emptyline=[]
for row in df['isi']:
    vs=analyzer.polarity_scores(row)
    emptyline.append(vs)
# Creating new dataframe with sentiments
df_sentiments=pd.DataFrame(emptyline)



# Merging the sentiments back to reviews dataframe
df_c = pd.concat([df.reset_index(drop=True), df_sentiments], axis=1)


import numpy as np

# Convert scores into positive and negetive sentiments using some threshold
df_c['Sentiment'] = np.where(df_c['compound'] >= 0 , 'Positive','Negative')
st.write('Hasil Twitter Sentiment Analysis')
st.dataframe(df_c)

#visualisasi bar dan pie chart
result=df_c['Sentiment'].value_counts()
st.write('Barchart')
st.bar_chart(result)
st.write('Piechart')
df = px.data.tips()
fig = px.pie(result, values='Sentiment')
st.write(fig)


pos = df_c.loc[df_c['Sentiment'] == 'Positive']
st.write('5 Tweet Paling Positif')
pos=pos.sort_values(by='compound', ascending=False)
pos.iloc[:5]



neg = df_c.loc[df_c['Sentiment'] == 'Negative']
st.write('5 Tweet Paling Negative')
neg=neg.sort_values(by='compound', ascending=True)
neg.iloc[:5]

from wordcloud import wordcloud
from wordcloud import WordCloud, STOPWORDS
st.set_option('deprecation.showPyplotGlobalUse', False)
wc=wordcloud.WordCloud(max_words=100, background_color="white", width = 500, height = 400).generate(' '.join(pos['isi']))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title('Hasil  Wordcloud Komentar Positif')
plt.show()
st.pyplot()

wc=wordcloud.WordCloud(max_words=100, background_color="white", width = 500, height = 400).generate(' '.join(neg['isi']))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title('Hasil Wordcloud Komentar Negatif')
plt.show()
st.pyplot()

