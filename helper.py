# Importing modules
import streamlit as st
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter

# Object
extract = URLExtract()

# -1 => Negative
# 0 => Neutral
# 1 => Positive

def fetch_stats(selected_user,d):

    if selected_user != 'Overall':
        d = d[d['user'] == selected_user]

    # fetch the number of messages
    num_messages = d.shape[0]

    # fetch the total number of words
    words = []
    for message in d['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = d[d['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in d['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

# Will return count of messages of selected user per day having k(0/1/-1) sentiment
def week_activity_map(selected_user,d,k):
    if selected_user != 'Overall':
        d = d[d['user'] == selected_user]
    d = d[d['value'] == k]
    return d['day_name'].value_counts()


# Will return count of messages of selected user per month having k(0/1/-1) sentiment
def month_activity_map(selected_user,d,k):
    if selected_user != 'Overall':
        d = d[d['user'] == selected_user]
    d = d[d['value'] == k]
    return d['month'].value_counts()

# Will return hear map containing count of messages having k(0/1/-1) sentiment
def activity_heatmap(selected_user,d,k):
    if selected_user != 'Overall':
        d = d[d['user'] == selected_user]
    d = d[d['value'] == k]
    
    # Creating heat map
    user_heatmap = d.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap


# Will return count of messages of selected user per date having k(0/1/-1) sentiment
def daily_timeline(selected_user,d,k):
    if selected_user != 'Overall':
        d = d[d['user'] == selected_user]
    d = d[d['value']==k]
    # count of message on a specific date
    daily_timeline = d.groupby('only_date').count()['message'].reset_index()
    return daily_timeline


# Will return count of messages of selected user per {year + month number + month} having k(0/1/-1) sentiment
def monthly_timeline(selected_user,d,k):
    if selected_user != 'Overall':
        d = d[d['user'] == selected_user]
    d = d[d['value']==-k]
    timeline = d.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

# Will return percentage of message contributed having k(0/1/-1) sentiment
def percentage(d,k):
    d = round((d['user'][d['value']==k].value_counts() / d[d['value']==k].shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return d

# Return wordcloud from words in message
def create_wordcloud(selected_user,d,k):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    if selected_user != 'Overall':
        d = d[d['user'] == selected_user]
    
    # Remove entries of no significance
    temp = d[d['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    
    # Remove stop words according to text file "stop_hinglish.txt"
    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)
    # Dimensions of wordcloud
    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    
    # Actual removing
    temp['message'] = temp['message'].apply(remove_stop_words)
    temp['message'] = temp['message'][temp['value'] == k]
    
    # Word cloud generated
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

# Return set of most common words having k(0/1/-1) sentiment
def most_common_words(selected_user,d,k):
    f = open('stop_hinglish.txt','r')
    stop_words = f.read()
    if selected_user != 'Overall':
        d = d[d['user'] == selected_user]
    temp = d[d['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    words = []
    for message in temp['message'][temp['value'] == k]:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
                
    # Creating data frame of most common 20 entries
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df