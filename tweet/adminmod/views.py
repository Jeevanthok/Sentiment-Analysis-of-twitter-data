from pathlib import Path
from sys import path
from django.shortcuts import render
from itertools import count
from operator import imod
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import re
from joblib import dump, load
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.http import HttpResponseForbidden
from account.models import Profile
from django.contrib import messages
from pathlib import Path
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Create your views here.
def index(request):
    if(request.session.has_key('account_id')):
        content = {}
        content['title'] = 'Welcome to twitter sentimental analysis'
        if(request.method == 'POST'):
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')

            df = pd.read_csv(str(BASE_DIR) + '/dataset/Twitter_Data.csv')
            print(df.head())
            inputs = df['clean_text'] # Independent X axis
            target = df['category'] # Dependent Y -> X

            inputs.fillna('', inplace=True)

            #Apply function on text column
            inputs = inputs.apply(cleaning)

            X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3) # 10000 -> Training = 7000 & Testing = 3000

            clf = Pipeline([('vectorizer', CountVectorizer()), ('nb', MultinomialNB())])
            clf.fit(X_train, y_train)
            dump(clf, str(BASE_DIR) + '/model/model.pkl')
            content['score'] = clf.score(X_test, y_test)
            messages.success(request, "Dataset trained and model.pkl updated.")
        return render(request, 'admin/train.html', content)
    else:
        messages.error(request, "Please login first.")
        return HttpResponseRedirect(reverse('account-login'))

#Removal of HTML Contents
def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removal of Punctuation Marks
def remove_punctuations(text):
    return re.sub('\[[^]]*\]', '', text)

# Removal of Special Characters
def remove_characters(text):
    return re.sub("[^a-zA-Z]", " ", text)

#Removal of stopwords
def remove_stopwords_and_lemmatization(text):
    final_text = []
    text = text.lower()
    text = nltk.word_tokenize(text)

    for word in text:
        if word not in set(stopwords.words('english')):
            lemma = nltk.WordNetLemmatizer()
            word = lemma.lemmatize(word)
            final_text.append(word)
    return " ".join(final_text)

#Total function
def cleaning(text):
    text = remove_html(text)
    text = remove_punctuations(text)
    text = remove_characters(text)
    text = remove_stopwords_and_lemmatization(text)
    return text
