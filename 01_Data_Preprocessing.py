# -*- coding: utf-8 -*-
"""
@author: Paula García Serrano

"""

import pandas as pd
import nltk
import re

def normalize_opinion(text):
    
    # import the english stop words list from NLTK
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    # Instantiate stemming class
    stemmer = nltk.stem.PorterStemmer() 
    
    # define emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    
    filtered_text = []
    
    # lowercase
    s = text.lower()
    
    #####-----change abreviations-----#####
    # Change 'n't to 'not'
    s = re.sub(r"n\'t", " not", s)
    s = re.sub(r"n\’t", " not", s)
    # Change ''s to 'is'
    s = re.sub(r"\'s", " is", s)
    s = re.sub(r"\’s", " is", s)
    # Change ''ll to 'will'
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"\’ll", " will", s)
    # Change ''m to 'am'
    s = re.sub(r"\'m", " am", s)
    s = re.sub(r"\’m", " am", s)
    # Change ''ve to 'have'
    s = re.sub(r"\'ve", " have", s)
    s = re.sub(r"\’ve", " have", s)
    # Change ''re to 'are'
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\’re", " are", s)
    
    #####-----remove special characters-----#####
    # Remove @name - tags/mentions
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Remove hashtags
    s = re.sub(r'#[A-Za-z0-9_]+', ' ', s)
    # Remove web links
    s = re.sub(r'http\S+', ' ', s)
    # Remove emojis
    s = emoji_pattern.sub(r'', s)
    
    #####-----remove punctuation-----#####
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    
    # Tokenization, removing stop words and stemming
    filtered_text = [stemmer.stem(w) for w in nltk.word_tokenize(s) if w not in stop_words]  
    s = ' '.join(filtered_text)
    

    # Remove words of length shorter than 2   
    s = [i for i in s.split() if len(i)>2]
    s = ' '.join(s)

    return  s

train = pd.read_csv('trainset.csv') # change the route to your train file
val = pd.read_csv('devset.csv') # change the route to your validation file

train['normalized_text'] = train['Text_data'].apply(lambda x: normalize_opinion(x))
val['normalized_text'] = val['Text data'].apply(lambda x: normalize_opinion(x))