# Librairies
import streamlit as st
import numpy as np
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

def process_text(text):
    # mise en forme du texte
    text_prep = transform_bow_fct(text)
    text_split = ["".join(word) for word in text_prep.split(" ")]
    final_text = [np.array(text_split, dtype='<U41')]
    return text_split, final_text
    
def fetch_tag(text):
    y_pred = pipe.predict(X_text)
    y_pred_inversed = mlb.inverse_transform(y_pred)
    return y_pred_inversed


def main():
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'

    # Chargement du vectorizer, multiLablbinarizer
    pipe = load_pipe("./models/pipeline.pkl")
    mlb = load_mlb("./models/mlb.pkl")
    
    st.title("Catégoriser automatiquement une question")

    input_text = st.text_input('Question')

    formatted_text, final_text = process_text(input_text)
    st.write('Texte formaté : ', formatted_text)

    if st.button('Rechercher les tags'):
#        json_data = json.dumps(final_text.tolist())
#        pred = request_prediction(MLFLOW_URI, json_data)[0] * 100000
#        pred_txt = fetch_tag(pred)
#        st.success(pred_txt, icon="✅")
        pred_txt = fetch_tag(final_text)
        st.success(pred_txt, icon="✅")

if __name__ == '__main__':
    main()