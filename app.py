# librairies
import streamlit as st
import numpy as np
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score

@st.cache_resource
def load_clf(add_clf):
    # recherche du classifieur pré-entrainé
    file_clf = open(add_clf, 'rb')
    clf = pickle.load(file_clf)
    return clf
 
@st.cache_resource
def load_vect(add_vect):   
    # recherche du vectoriseur pré-entrainé
    file_vect = open(add_vect, 'rb')
    vect = pickle.load(file_vect)
    return vect

@st.cache_resource
def load_mlb(add_mlb):
     # recherche du multiLabelBinarizer pré-entrainé
    file_mlb = open(add_mlb, 'rb')
    mlb = pickle.load(file_mlb)
    return mlb
    
# Tokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

def tokenizer_fct(sentence) :
    # print(sentence)
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').\
                             replace('#', ' ').replace('<p>', ' ').replace('>', ' ').replace('<', ' ')
    # Remove ponctuation (except # and ++ for c# and c++)
    sentence_clean1 = re.sub('[^\\w\\s#\\s++]', ' ', sentence_clean)

    # Remove numbers
    sentence_clean2 = re.sub(r'\w*\d+\w*', ' ', sentence_clean1)

    # Remove extra spaces
    sentence_clean = re.sub('\s+', ' ', sentence_clean2)
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens

# Stop words
from nltk.corpus import stopwords
stop_w = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')']
stop_w.extend(['code', 'quot', 'use', 'http', 'com', 'error', 'work', 'want', 'one', 'would', 'need', 
                   'help', 'also', 'exampl', 'could', 'thing', 'well', 'dear', 'p'])

def stop_word_filter_fct(list_words) :
    filtered_w = [w for w in list_words if not w in stop_w]
#    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w

# lower case et alpha
def lower_start_fct(list_words) :
    lw = [w.lower() for w in list_words if (not w.startswith("@")) 
                                      and (not w.startswith("#"))
                                       and (not w.startswith("http"))]
    return lw

# Lemmatizer (base d'un mot)
from nltk.stem import WordNetLemmatizer

def lemma_fct(list_words) :
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w

# Fonction de préparation du texte pour le bag of words (Countvectorizer et Tf_idf, Word2Vec)
def transform_bow_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    # lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lw)
    return transf_desc_text
#    return lw

def process_text(text):
    # mise en forme du texte
    text_prep = transform_bow_fct(text)
    tt = ""
    for word in text_prep.split(" "):
        tt = tt + word + " "
    text_final = []
    text_final.append(tt)
    return text_final
    
def fetch_tag(text):
    X_text = vect.transform(text)    
    y_pred = clf.predict(X_text)
    y_pred_inversed = mlb.inverse_transform(y_pred)
    return y_pred_inversed

st.title("Catégoriser automatiquement une question")

clf = load_clf("./models/clf.pkl")
vect = load_vect("./models/vect.pkl")
mlb = load_mlb("./models/mlb.pkl")

input_text = st.text_input('Question')

formatted_text = process_text(input_text)
st.write('Texte mise en forme : ', formatted_text)

if st.button('Rechercher les tags'):
    y_pred_inversed = fetch_tag(formatted_text) 
    st.write('Tags proposés : ', y_pred_inversed)
    