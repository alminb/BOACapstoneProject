import os
import pandas as pd
from datetime import date
from flask import Flask,request,render_template,session,redirect,url_for,flash

import matplotlib.pyplot as plt
import numpy as np
import re
import nltk

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

app = Flask(__name__)


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

@app.route('/results',methods=['POST','GET'])
def results():
    if request.method == 'POST':
        #Twitter Crawler Start ---------------------------------------------------------------------
        print("In POST")
        start_day = date.today()
        end_date = start_day
        print(end_date)
        twitterHashtag = request.form['twitterHashtag']
        search_term = "#"+twitterHashtag
        from_date = request.form['from_date']
        print("from_date: ", from_date)
        os.system(f"snscrape --since {from_date} twitter-search '{search_term}' > result-tweets.txt")
        if os.stat("result-tweets.txt").st_size == 0:
            counter = 0
        else:
            df = pd.read_csv('result-tweets.txt',names=['link'])
            counter = df.size
        print('Number of Tweets : ' + str(counter))
        max_results = 100
        extracted_tweets = "snscrape --format ~{content!r}~" + f" --max-results {max_results} --since {from_date} twitter-search '{search_term}' > extracted-tweets.txt"
        os.system(extracted_tweets)
        tweets=[]
        if os.stat("extracted-tweets.txt").st_size == 0:
            print("No tweets found")
        else:
            df = pd.read_csv('extracted-tweets.txt',delimiter="~~",skipinitialspace=True, names=['content'])
            for row in df['content'].iteritems():
                print(row[1][1:-1])
                tweets.append(row[1][1:-1])
        # Twitter Crawler End ---------------------------------------------------------------------

        # Twitter Processor Start -----------------------------------------------------------------
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('omw-1.4')
        nltk.download('vader_lexicon')


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
            print(row1)
            cleanedTweets.append(row1[1])
        # Twitter Processor End -------------------------------------------------------------------
        # Sentiment Analysis  -------------------------------------
        sia = SentimentIntensityAnalyzer()
        df["sentiment_score2"] = df["New_cleanText"].apply(lambda x: sia.polarity_scores(x)["compound"])
        df["sentiment2"] = np.select([df["sentiment_score2"] < 0, df["sentiment_score2"] == 0, df["sentiment_score2"] > 0],
                                    ['negative', 'neutral', 'positive'])
        cleanedSentiments=[]
        for row1 in df['sentiment2'].iteritems():
            print(row1)
            cleanedSentiments.append(row1[1])

        # Sentiment Analysis ---------------------------------------
        return render_template("results.html",tweets=tweets,cleanedTweets=cleanedTweets, cleanedSentiments=cleanedSentiments)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")


if __name__ == '__main__':
    app.run()
