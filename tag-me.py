# Librairies
import streamlit as st
import numpy as np
from PIL import Image
import re
import pickle
import requests
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import words

nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')

@st.cache_resource
# Chargement du Vectorizer 
def load_pipe(add_pipe):   
    file_pipe = open(add_pipe, 'rb')
    pipe = pickle.load(file_pipe)
    return pipe

@st.cache_resource
# Chargement du multiLabelBinarizer pré-entrainé
def load_mlb(add_mlb): 
    file_mlb = open(add_mlb, 'rb')
    mlb = pickle.load(file_mlb)
    return mlb

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'Body': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

# Tokenizer
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

# Fonction de mise en forme et nettoyage
def process_text(text):
    text_prep = transform_bow_fct(text)
    text_split = ["".join(word) for word in text_prep.split(" ")]
    final_text = [np.array(text_split, dtype='<U41')]
    return text_split, final_text


def main():
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'

    # Chargement du vectorizer, multiLablbinarizer
    pipe = load_pipe("./models/pipeline.pkl")
    mlb = load_mlb("./models/mlb.pkl")
    
    image = Image.open('logo.jpg')
    st.image(image, width=50)
    st.title(":orange[Tag-Me]")
    st.header("Catégoriser automatiquement une question")

    input_text = st.text_input('Poser votre question')

    formatted_text, final_text = process_text(input_text)
    st.write('Texte formaté : ', formatted_text)

    if st.button('Rechercher les tags'):
        y_pred = pipe.predict(final_text)
        if final_text != "":
            y_pred_inversed = mlb.inverse_transform(y_pred)
            st.success(y_pred_inversed, icon="✅")
        else:
            st.error('Tags inexistants', icon="🚨")

if __name__ == '__main__':
    main()