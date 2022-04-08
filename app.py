import os
import pandas as pd
from datetime import date
from datetime import timedelta
from flask import Flask,request,render_template,session,redirect,url_for,flash

import matplotlib.pyplot as plt
import numpy as np
import re
import nltk
from langdetect import detect

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, plot_confusion_matrix
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import RegexpTokenizer

from nltk.sentiment import SentimentIntensityAnalyzer
import operator

from pygooglenews import GoogleNews
import newspaper
import json
import time

app = Flask(__name__)

nltk.download('gutenberg')

@app.route('/')
def entry():
    today_date = date.today()
    print(today_date)
    return render_template("home.html", today_date=today_date)

@app.route('/home')
def home():
    today_date = date.today()
    print(today_date)
    return render_template("home.html", today_date=today_date)

@app.route('/resultsTwitter',methods=['POST','GET'])
def resultsTwitter():
    if request.method == 'POST':
        #Twitter Crawler Start ---------------------------------------------------------------------
        twitterHashtag = request.form['twitterHashtag']
        search_term = "#"+twitterHashtag

        if 'from_date'in request.form:  #check if start date provided, else use yesterdays's date
            if request.form['from_date']!='':
                from_date = request.form['from_date']
            else:
                today = date.today()
                from_date = today-timedelta(days=1)

        max_results = 100
        extracted_tweets = "snscrape --format ~{content!r}~" + f" --max-results {max_results} --since {from_date}  twitter-search '{search_term}' > extracted-tweets.txt"
        os.system(extracted_tweets)
        tweets=[]
        if os.stat("extracted-tweets.txt").st_size == 0:
            print("No tweets found")
        else:
            df = pd.read_csv('extracted-tweets.txt',delimiter="~~",skipinitialspace=True, names=['content'])
            rowCounter=0
            for row in df['content'].items():
                if detect(row[1][1:-1])=='en':
                    tweets.append(row[1][1:-1])
                else:
                    df.drop(labels=[rowCounter],axis=0,inplace=True)
                rowCounter+=1;
        # Twitter Crawler End ---------------------------------------------------------------------

        # Twitter Processor Start -----------------------------------------------------------------
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('omw-1.4')
        nltk.download('vader_lexicon')
        nltk.download('gutenberg')


        # Twitter Processor CONTD -----------------------------------------------------------------
        STOPWORDS = stopwords.words('english')
        df['clean_text'] = df['content'].apply(lambda x: re.sub(r'\w*\d\w*', '', x).strip())
        df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([w for w in x.split(' ') if w not in STOPWORDS]))
        content = df['clean_text'].values.tolist()
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()

        def preprocess(sentence):
            sentence = str(sentence)
            sentence = sentence.lower()
            sentence = sentence.replace('{html}', "")
            cleanr = re.compile('<.*?>')
            cleantext = re.sub(cleanr, '', sentence)
            rem_url = re.sub(r'http\S+', '', cleantext)
            rem_num = re.sub('[0-9]+', '', rem_url)
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(rem_num)
            filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
            stem_words = [stemmer.stem(w) for w in filtered_words]
            lemma_words = [lemmatizer.lemmatize(w) for w in stem_words]
            return " ".join(filtered_words)

        cleanedTweets=[]
        df['New_cleanText'] = df['clean_text'].map(lambda s: preprocess(s))
        for row1 in df['New_cleanText'].iteritems():
            cleanedTweets.append(row1[1])
        # Twitter Processor End -------------------------------------------------------------------
        # Sentiment Analysis  -------------------------------------
        sia = SentimentIntensityAnalyzer()
        df["sentiment_score2"] = df["content"].apply(lambda x: sia.polarity_scores(x)["compound"])
        df["sentiment2"] = np.select([df["sentiment_score2"] < 0, df["sentiment_score2"] == 0, df["sentiment_score2"] > 0],
                                    ['negative', 'neutral', 'positive'])
        cleanedSentiments=[]
        for row1 in df['sentiment2'].iteritems():
            cleanedSentiments.append(row1[1])

        # Sentiment Analysis ---------------------------------------
        return render_template("results.html",tweets=tweets,cleanedTweets=cleanedTweets, cleanedSentiments=cleanedSentiments)

#results when using news article crawler only
@app.route('/resultsNews',methods=['POST','GET'])
def resultsNews():
    if request.method=='POST':
        companyName = request.form['companyName']
        from_date=request.form['from_date']

        gn = GoogleNews()
        newsArticles = gn.search(companyName)
        newsArticles = newsArticles['entries']
        articles = []
        for entry in newsArticles:
            url = entry['link']
            article = newspaper.Article(url=url, language='en')
            article.download()
            try:
                article.parse();
                article = {
                    "title": str(article.title),
                    "text": str(article.text),
                    "authors": article.authors,
                    "published_date": str(article.publish_date),
                    "top_image": str(article.top_image),
                    "videos": article.movies,
                    "keywords": article.keywords,
                    "summary": str(article.summary)
                }
                #print("----------"+article["title"] + "------:" + article["text"] + "\n\n")
                articles.append(article);
            except: #if url can not be parsed by parser go to next entry
                continue

        return render_template("resultsNews.html",articles=articles)


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")


if __name__ == '__main__':
    app.run()
