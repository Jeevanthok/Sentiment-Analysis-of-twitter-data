from os import stat
import tweepy
from django.contrib import messages
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.http import HttpResponseForbidden
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from joblib import dump, load
from pathlib import Path

from tweepy.cursor import Cursor
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Create your views here.

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

def index(request):
    if(request.session.has_key('account_id')):
        content = {}
        content['title'] = 'Welcome to twitter sentimental analysis'
        content['tweet'] = ''
        if(request.method == 'POST'):
            model= load(str(BASE_DIR) + '/model/model.pkl')
            tweet = request.POST['tweet']
            content['tweet'] = tweet
            tweet = cleaning(tweet)
            content['prediction'] = model.predict([tweet])
        return render(request, 'home/index.html', content)
    else:
        messages.error(request, "Please login first.")
        return HttpResponseRedirect(reverse('account-login'))


def livetweets(request):
    if(request.session.has_key('account_id')):
        content = {}
        content['title'] = 'Live Tweets'
        if(request.method == 'POST'):
            # Twitter Developer keys here
            consumer_key = 'VrH8rSQ7e6PyJf8qTeUSsGMyw'
            consumer_key_secret = '8oj2W45fsU4T2WmQgIDLhAOMdiXVIXk4VZbu8jUr48JSD8wjK0'
            access_token = '2895910574-93iYmkVhx6z3yNnP4dSvjQALAiLUI9wfbVx053I'
            access_token_secret = 'D0dDR9sUaEPajD8iKFx1KuDn1SIJBjkGNEVIctWIB38J4'

            auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
            auth.set_access_token(access_token, access_token_secret)
            api = tweepy.API(auth)

            # Fetch tweets
            tweet_id = request.POST['userid']
            no_tweets = int(request.POST['tweets_no'])

            user = api.get_user(tweet_id)
            print(user.statuses_count)

            tweet_count = 0
            tweet_list = []

            pos = 0
            neg = 0
            neu = 0

            model = load(str(BASE_DIR) + '/model/model.pkl')
            for status in  tweepy.Cursor(api.user_timeline, id = tweet_id).items():
                if(tweet_count >= no_tweets):
                    break
                else:
                    tweet_count = tweet_count +  1
                    clean_status = cleaning(status.text)
                    predict = model.predict([clean_status])
                    if(predict == 1):
                        pos = pos + 1
                    elif(predict == 0):
                        neu = neu + 1
                    else:
                        neg = neg + 1
                    tweet_dict = {'text': status.text, 'predict': predict}
                    tweet_list.append(tweet_dict)

            content['tweet_list'] = tweet_list
            content['pos'] = pos
            content['neg'] = neg
            content['neu'] = neu
            content['user'] = user.screen_name
            content['tweet_count'] = str(no_tweets)

        return render(request, 'home/live.html', content)
    else:
        messages.error(request, "Please login first.")
        return HttpResponseRedirect(reverse('account-login'))

def about(request):
    content = {}
    content['title'] = 'About'
    return render(request, 'about.html', content)

def team(request):
    content = {}
    content['title'] = 'team'
    return render(request, 'home/team.html', content)


