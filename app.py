# Importing modules
import nltk
import streamlit as st
import re
import preprocessor,helper,senti
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# App headere
st.sidebar.header("Whatsapp Chat Analyzer")

# File upload button
uploaded_file = st.sidebar.file_uploader("Choose a file")

# Main heading
st. markdown("<h1 style='text-align: center; color: white;'>Whatsapp Chat Sentiment Analyzer</h1>", unsafe_allow_html=True)

if uploaded_file is not None:
    
    # Getting byte form & then decoding
    bytes_data = uploaded_file.getvalue()
    d = bytes_data.decode("utf-8")
    
    # Perform preprocessing
    data = preprocessor.preprocess(d)
    data=senti.sentiment_analysis(data)
    # Importing SentimentIntensityAnalyzer class from "nltk.sentiment.vader"
    # from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    # # Object
    # sentiments = SentimentIntensityAnalyzer()
    
    # # Creating different columns for (Positive/Negative/Neutral)
    # data["po"] = [sentiments.polarity_scores(i)["pos"] for i in data["message"]] # Positive
    # data["ne"] = [sentiments.polarity_scores(i)["neg"] for i in data["message"]] # Negative
    # data["nu"] = [sentiments.polarity_scores(i)["neu"] for i in data["message"]] # Neutral
    
    # To indentify true sentiment per row in message column
    # from keras.models import load_model
    # best_model = load_model('best_model2.hdf5')
    # sentiments=["neutral","negetive","positive"]
    # tokenizer = Tokenizer(num_words=5000)
    # exg=[]
    # print("jnjn")
    # for str in data["message"]:
    #     sequence = tokenizer.texts_to_sequences([str])
    #     test = pad_sequences(sequence, maxlen=200)
    #     exg.append(sentiments[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]])
    # data['sentiment']=exg
    # def sentiment(d):
    #     if d["sentiment"] == "positive":
    #         return 1
    #     elif d["sentiment"] == "negetive":
    #         return -1
    #     else:
    #         return 0

    # # Creating new column & Applying function
    # data['value'] = data.apply(lambda row: sentiment(row), axis=1)
    
    # User names list
    user_list = data['user'].unique().tolist()
    
    # Sorting
    user_list.sort()
    
    # Insert "Overall" at index 0
    user_list.insert(0, "Overall")
    
    # Selectbox
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)
    
    if st.sidebar.button("Show Analysis"):
        
        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,data)
        st.header("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.subheader("Total Messages")
            st.subheader(num_messages)
        with col2:
            st.subheader("Total Words")
            st.subheader(words)
        with col3:
            st.subheader("Media Shared")
            st.subheader(num_media_messages)
        with col4:
            st.subheader("Links Shared")
            st.subheader(num_links)
            
        # Monthly activity map
        st.header("Monthly activity map")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h5 style='text-align: center;'>Monthly Activity map(Positive)</h5>",unsafe_allow_html=True)
            
            busy_month = helper.month_activity_map(selected_user, data,1)
            
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h5 style='text-align: center;'>Monthly Activity map(Neutral)</h5>",unsafe_allow_html=True)
            
            busy_month = helper.month_activity_map(selected_user, data, 0)
            
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h5 style='text-align: center;'>Monthly Activity map(Negative)</h5>",unsafe_allow_html=True)
            
            busy_month = helper.month_activity_map(selected_user, data, -1)
            
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Daily activity map
        st.header("Daily activity map")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h5 style='text-align: center;'>Daily Activity map(Positive)</h5>",unsafe_allow_html=True)
            
            busy_day = helper.week_activity_map(selected_user, data,1)
            
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h5 style='text-align: center;'>Daily Activity map(Neutral)</h5>",unsafe_allow_html=True)
            
            busy_day = helper.week_activity_map(selected_user, data, 0)
            
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h5 style='text-align: center;'>Daily Activity map(Negative)</h5>",unsafe_allow_html=True)
            
            busy_day = helper.week_activity_map(selected_user, data, -1)
            
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)


        # Daily timeline
        st.header("Daily Timeline")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h5 style='text-align: center;'>Daily Timeline(Positive)</h5>",unsafe_allow_html=True)
            
            daily_timeline = helper.daily_timeline(selected_user, data, 1)
            
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h5 style='text-align: center;'>Daily Timeline(Neutral)</h5>",unsafe_allow_html=True)
            
            daily_timeline = helper.daily_timeline(selected_user, data, 0)
            
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h5 style='text-align: center;'>Daily Timeline(Negative)</h5>",unsafe_allow_html=True)
            
            daily_timeline = helper.daily_timeline(selected_user, data, -1)
            
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Monthly timeline
        st.header("Monthly Timeline")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h5 style='text-align: center;'>Monthly Timeline(Positive)</h5>",unsafe_allow_html=True)
            
            timeline = helper.monthly_timeline(selected_user, data,1)
            
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h5 style='text-align: center;'>Monthly Timeline(Neutral)</h5>",unsafe_allow_html=True)
            
            timeline = helper.monthly_timeline(selected_user, data,0)
            
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h5 style='text-align: center;'>Monthly Timeline(Negative)</h5>",unsafe_allow_html=True)
            
            timeline = helper.monthly_timeline(selected_user, data,-1)
            
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Percentage contributed
        st.header("Percentage contributed")
        if selected_user == 'Overall':
            col1,col2,col3 = st.columns(3)
            with col1:
                st.markdown("<h5 style='text-align: center;'>Most Positive Contribution</h5>",unsafe_allow_html=True)
                x = helper.percentage(data, 1)
                
                # Displaying
                st.dataframe(x)
            with col2:
                st.markdown("<h5 style='text-align: center;'>Most Neutral Contribution</h5>",unsafe_allow_html=True)
                y = helper.percentage(data, 0)
                
                # Displaying
                st.dataframe(y)
            with col3:
                st.markdown("<h5 style='text-align: center;'>Most Negative Contribution</h5>",unsafe_allow_html=True)
                z = helper.percentage(data, -1)
                
                # Displaying
                st.dataframe(z)


        # Most Positive,Negative,Neutral User...
        st.header("Most Positive,Negative,Neutral Users")
        if selected_user == 'Overall':
            
            # Getting names per sentiment
            x = data['user'][data['value'] == 1].value_counts().head(10)
            y = data['user'][data['value'] == -1].value_counts().head(10)
            z = data['user'][data['value'] == 0].value_counts().head(10)

            col1,col2,col3 = st.columns(3)
            with col1:
                # heading
                st.markdown("<h5 style='text-align: center;'>Most Positive Users</h5>",unsafe_allow_html=True)
                
                # Displaying
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                # heading
                st.markdown("<h5 style='text-align: center;'>Most Neutral Users</h5>",unsafe_allow_html=True)
                
                # Displaying
                fig, ax = plt.subplots()
                ax.bar(z.index, z.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                # heading
                st.markdown("<h5 style='text-align: center;'>Most Negative Users</h5>",unsafe_allow_html=True)
                
                # Displaying
                fig, ax = plt.subplots()
                ax.bar(y.index, y.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

        # WORDCLOUD......
        st.header("WordCloud Most Commonly Used Words")
        col1,col2,col3 = st.columns(3)
        with col1:
            try:
                # heading
                st.markdown("<h5 style='text-align: center;'>Positive WordCloud</h5>",unsafe_allow_html=True)
                
                # Creating wordcloud of positive words
                df_wc = helper.create_wordcloud(selected_user, data,1)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                # # Display error message
                st.image('error.webp')
        with col2:
            try:
                # heading
                st.markdown("<h5 style='text-align: center;'>Neutral WordCloud</h5>",unsafe_allow_html=True)
                
                # Creating wordcloud of neutral words
                df_wc = helper.create_wordcloud(selected_user, data,0)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')
        with col3:
            try:
                # heading
                st.markdown("<h5 style='text-align: center;'>Negative WordCloud</h5>",unsafe_allow_html=True)
                
                # Creating wordcloud of negative words
                df_wc = helper.create_wordcloud(selected_user, data,-1)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')

        # Most common positive words
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                # Data frame of most common positive words.
                most_common_df = helper.most_common_words(selected_user, data,1)
                
                # heading
                st.markdown("<h5 style='text-align: center;'>Positive Words</h5>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1],color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')
        with col2:
            try:
                # Data frame of most common neutral words.
                most_common_df = helper.most_common_words(selected_user, data,0)
                
                # heading
                st.markdown("<h5 style='text-align: center;'>Neutral Words</h5>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1],color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')
        with col3:
            try:
                # Data frame of most common negative words.
                most_common_df = helper.most_common_words(selected_user, data,-1)
                
                # heading
                st.markdown("<h5 style='text-align: center;'>Negative Words</h5>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')



