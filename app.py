import csv
import os
import string
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

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


from nltk.sentiment import SentimentIntensityAnalyzer
import operator

from pygooglenews import GoogleNews
import newspaper
import json
import time
import wordcloud

app = Flask(__name__)

nltk.download('gutenberg')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def maxi(x):
    maximum = x[0]
    for i in range (0,len(x)):
        if x[i] >= maximum:
            maximum = x[i]
            index=i
    return index

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
        #-----------------------------------------------------------Twitter Crawler Start-----------------------------------------------------------
        twitterHashtag = request.form['twitterHashtag']
        search_term = "#"+twitterHashtag

        if 'from_date'in request.form:  #check if start date provided, else use yesterdays's date
            if request.form['from_date']!='':
                from_date = request.form['from_date']
            else:
                today = date.today()
                from_date = today-timedelta(days=1)

        max_results = 1000
        extracted_tweets = "snscrape --format ~{date}~~{content!r}~" + f" --max-results {max_results} --since {from_date}  twitter-hashtag '{search_term}' > extracted-tweets.txt"
        os.system(extracted_tweets)
        tweets=[]
        if os.stat("extracted-tweets.txt").st_size == 0:
            print("No tweets found")
        else:
            df = pd.read_csv('extracted-tweets.txt',delimiter="~~",skipinitialspace=True, names=['date','content'])
            rowCounter=0
            for row in df['content'].items():
                if detect(row[1][1:-1])=='en':
                    tweets.append(row[1][1:-1])
                else:
                    df.drop(labels=[rowCounter],axis=0,inplace=True)
                rowCounter+=1;
        #-----------------------------------------------------------Twitter Crawler End-----------------------------------------------------------

        #-----------------------------------------------------------Twitter Processor Start-----------------------------------------------------------
        stop = stopwords.words('english')
        STOPWORDS = stopwords.words('english')
        df['stopwords'] = df['content'].apply(lambda x: len([x for x in str(x).split() if x in stop]))
        df['content'] = df['content'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
        df['content'] = df['content'].str.replace('[^\w\s]','')
        df['content'] = df['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
        def remove_url(text):
            url = re.compile(r'https?://\S+|www\.\S+')
            return url.sub(r'', text)
        df['content'] = df['content'].apply(lambda x: remove_url(x))
        def remove_html(text):
            html = re.compile(r'<.*?>')
            return html.sub(r'', text)
        df['content'] = df['content'].apply(lambda x: remove_html(x))
        def remove_emoji(text):
            emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags 
                                       u"\U00002702-\U000027B0"
                                       u"\U000024C2-\U0001F251"
                                       "]+", flags=re.UNICODE)
            return emoji_pattern.sub(r'', text)
        # remove all emojis from df
        df['content'] = df['content'].apply(lambda x: remove_emoji(x))

        def clean_text_round1(text):
            '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
            text = text.lower()
            text = re.sub('\[.*?\]', '', text)
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub('\w*\d\w*', '', text)
            return text

        round1 = lambda x: clean_text_round1(x)
        df['content'] = df.content.apply(round1)

        def clean_text_round2(text):
            text = re.sub('[‘’“”…]', '', text)
            text = re.sub('\n', '', text)
            return text

        round2 = lambda x: clean_text_round2(x)
        df['content'] = df.content.apply(round2)

        cleanedTweets=[]
        for row1 in df['content'].iteritems():
            cleanedTweets.append(row1[1])

        # TF-IDF
        STOPWORDS = stopwords.words('english')
        vect = TfidfVectorizer(stop_words=STOPWORDS, max_features=2000)
        vect_text = vect.fit_transform(df['content'])
        # LDA
        lda_model = LatentDirichletAllocation(n_components=30,
                                              learning_method='online', random_state=100, max_iter=20)
        lda_top = lda_model.fit_transform(vect_text)
        # find top 10 words in each topic
        vocab = vect.get_feature_names()
        topics = []
        topicsKey = []
        for i, comp in enumerate(lda_model.components_):
            topicsKey.append("")
            vocab_comp = zip(vocab, comp)
            sorted_words = sorted(vocab_comp, key=lambda x: x[1], reverse=True)[:10]
            c, v = zip(*sorted_words)
            topics.append(c)
            for t in sorted_words:
                topicsKey[i]+=(t[0]+", ")
        topic_index = []
        topicofTweet=[]
        for i, topic in enumerate(lda_top):
            print("tweet ",i,": ",maxi(topic))
            topic_index.append(maxi(topic))
            topicofTweet.append(""+str(maxi(topic)))
        df['topic_index'] = topic_index
        topicsampleCount = {}
        for x in topicofTweet:
            if x not in topicsampleCount:
                topicsampleCount[x] = 1
            else:
                topicsampleCount[x] += 1
        topicsampleCount = sorted(topicsampleCount.items(), key=lambda x: x[1], reverse=True)

        X = df['content'].values
        y = df['topic_index'].values.reshape(-1, 1)

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Random Forest for tweets
        from sklearn.ensemble import RandomForestClassifier
        forest = RandomForestClassifier(n_estimators=123, random_state=18, max_depth=300)
        forest.fit(X_train, y_train)
        y_pred_forest = forest.predict(X_test)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred_forest))

        #-----------------------------------------------------------Twitter Processor End-----------------------------------------------------------


        #-----------------------------------------------------------Sentiment Analysis start-----------------------------------------------------------
        sia = SentimentIntensityAnalyzer()
        df["sentiment_score2"] = df["content"].apply(lambda x: sia.polarity_scores(x)["compound"])
        df["sentiment2"] = np.select([df["sentiment_score2"] < 0, df["sentiment_score2"] == 0, df["sentiment_score2"] > 0],
                                    ['negative', 'neutral', 'positive'])
        cleanedSentiments=[]
        for row1 in df['sentiment2'].iteritems():
            cleanedSentiments.append(row1[1])
        dates=[]
        for row in df['date'].iteritems():
            dates.append(row[1][1:11])

        # keyword cloud
        w = wordcloud.WordCloud(background_color="rgba(255, 255, 255, 0)", mode="RGBA")

        def newsfunction(a):
            res = str()
            for i in a:
                res += ','
                res += i
            return ''.join(res)

        freq = pd.Series(' '.join(df['content']).split()).value_counts()[:50]
        new_data = pd.DataFrame(freq)
        new_data['keywords'] = freq.index
        abc = newsfunction(new_data['keywords'])
        w.generate(abc)
        w.to_file("static/pywordcloud1.png")


        #-----------------------------------------------------------Sentiment Analysis end-----------------------------------------------------------
        return render_template("results.html",tweets=tweets,topicsKey= topicsKey,topicsofTweets=topicofTweet,topicCount=topicsampleCount,cleanedTweets=cleanedTweets, cleanedSentiments=cleanedSentiments,tweetDates=dates)

#results when using news article crawler only
@app.route('/resultsNews',methods=['POST','GET'])
def resultsNews():
    if request.method=='POST':
        print("Starting News Crawler")
        companyName = request.form['companyName']
        from_date=request.form['from_date']

        #-----------------------------------------------------------NEWS CRAWLER START-----------------------------------------------------------
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
                if str(article.title)=="Are you a robot?":
                    continue
                if str(article.text)=="":
                    continue
                article = {
                    "title": str(article.title),
                    "text": str(article.text),
                    "authors": article.authors,
                    "published_date": str(article.publish_date)[0:10],
                    "top_image": str(article.top_image),
                    "videos": article.movies,
                    "keywords": article.keywords,
                    "summary": str(article.summary)
                }
                #print("----------"+article["title"] + "------:" + article["text"] + "\n\n")
                articles.append(article);
            except: #if url can not be parsed by parser go to next entry
                continue
        # -----------------------------------------------------------NEWS CRAWLER END-----------------------------------------------------------

        # article topic processor
        STOPWORDS = stopwords.words('english')
        dfa = pd.DataFrame(articles)
        dfa.head()

        #-----------------------------------------------------------SENTIMENT-----------------------------------------------------------
        cleanedSentiments = []
        sia = SentimentIntensityAnalyzer()
        dfa["sentiment_score2"] = dfa["text"].apply(lambda x: sia.polarity_scores(x)["compound"])
        dfa["sentiment2"] = np.select(
            [dfa["sentiment_score2"] < 0, dfa["sentiment_score2"] == 0, dfa["sentiment_score2"] > 0],
            ['negative', 'neutral', 'positive'])
        cleanedSentiments = []
        for row1 in dfa['sentiment2'].iteritems():
            cleanedSentiments.append(row1[1])
        #-----------------------------------------------------------SENTIMENT-----------------------------------------------------------

        #-----------------------------------------------------------TOPIC ANALYZER START-----------------------------------------------------------
        vect = TfidfVectorizer(stop_words=STOPWORDS, max_features=1000)
        vect_text = vect.fit_transform(dfa['text'])

        lda_model = LatentDirichletAllocation(n_components=30,
                                              learning_method='online', random_state=100, max_iter=20)
        lda_top = lda_model.fit_transform(vect_text)

        # find top 10 words in each topic
        vocab = vect.get_feature_names()
        topics = []
        topicsKey = []
        for i, comp in enumerate(lda_model.components_):
            topicsKey.append("")
            vocab_comp = zip(vocab, comp)
            sorted_words = sorted(vocab_comp, key=lambda x: x[1], reverse=True)[:10]
            c, v = zip(*sorted_words)
            topics.append(c)
            print(" ")
            print("Topic " + str(i) + ": ")
            for t in sorted_words:
                #             topics[i].append(t[0])
                topicsKey[i]+=(t[0]+", ")
                print(t[0], end=" ")
            print("\n")
        print(topicsKey)
        # find the index of the topic list of each articel.
        topicofArticle=[]
        topic_index = []
        for i, topic in enumerate(lda_top):
            #     print("Document i: ")
            print("article ", i, ": ", maxi(topic))
            topic_index.append(maxi(topic))
            topicofArticle.append(""+str(maxi(topic)))

        topicsampleCount={}
        for x in topicofArticle:
            if x not in topicsampleCount:
                topicsampleCount[x]=1
            else:
                topicsampleCount[x]+=1
        topicsampleCount = sorted(topicsampleCount.items(), key=lambda x: x[1], reverse=True)

        print(topicofArticle)
        dfa['topic_index'] = topic_index
        X = dfa['text'].values
        y = dfa['topic_index'].values.reshape(-1, 1)

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(X)
        # print(X[0][0])
        #-----------------------------------------------------------TOPIC ANALYZER END-----------------------------------------------------------
        #-----------------------------------------------------------KEYWORD CLOUD START-----------------------------------------------------------
        #keyword cloud
        w = wordcloud.WordCloud(background_color="rgba(255, 255, 255, 0)", mode="RGBA")

        def newsfunction(a):
            res = str()
            for i in a:
                res += ','
                res += i
            return ''.join(res)

        freq = pd.Series(' '.join(dfa['text']).split()).value_counts()[:50]
        new_data = pd.DataFrame(freq)
        new_data['keywords'] = freq.index
        abc = newsfunction(new_data['keywords'])
        w.generate(abc)
        w.to_file("static/pywordcloud2.png")
        # -----------------------------------------------------------KEYWORD CLOUD END-----------------------------------------------------------

        return render_template("resultsNews.html",articles=articles,topicsKey= topicsKey,topicsofArticle=topicofArticle,topicCount=topicsampleCount, sentiments=cleanedSentiments)


@app.route('/resultsBoth',methods=['POST','GET'])
def resultsBoth():
    if request.method=='POST':
        # Twitter Crawler Start ---------------------------------------------------------------------
        twitterHashtag = request.form['twitterHashtag']
        search_term = "#" + twitterHashtag

        if 'from_date' in request.form:  # check if start date provided, else use yesterdays's date
            if request.form['from_date'] != '':
                from_date = request.form['from_date']
            else:
                today = date.today()
                from_date = today - timedelta(days=1)

        max_results = 200
        extracted_tweets = "snscrape --format ~{date}~~{content!r}~" + f" --max-results {max_results} --since {from_date}  twitter-hashtag '{search_term}' > extracted-tweets.txt"
        os.system(extracted_tweets)
        tweets = []
        if os.stat("extracted-tweets.txt").st_size == 0:
            print("No tweets found")
        else:
            df = pd.read_csv('extracted-tweets.txt', delimiter="~~", skipinitialspace=True,
                             names=['date', 'content'])
            rowCounter = 0
            for row in df['content'].items():
                if detect(row[1][1:-1]) == 'en':
                    tweets.append(row[1][1:-1])
                else:
                    df.drop(labels=[rowCounter], axis=0, inplace=True)
                rowCounter += 1;
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
        df['clean_text'] = df['clean_text'].apply(
            lambda x: ' '.join([w for w in x.split(' ') if w not in STOPWORDS]))
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

        cleanedTweets = []
        df['New_cleanText'] = df['clean_text'].map(lambda s: preprocess(s))
        for row1 in df['New_cleanText'].iteritems():
            cleanedTweets.append(row1[1])
        # Twitter Processor End -------------------------------------------------------------------
        # Sentiment Analysis  -------------------------------------
        sia = SentimentIntensityAnalyzer()
        df["sentiment_score2"] = df["content"].apply(lambda x: sia.polarity_scores(x)["compound"])
        df["sentiment2"] = np.select(
            [df["sentiment_score2"] < 0, df["sentiment_score2"] == 0, df["sentiment_score2"] > 0],
            ['negative', 'neutral', 'positive'])
        cleanedSentiments = []
        for row1 in df['sentiment2'].iteritems():
            cleanedSentiments.append(row1[1])
        dates = []
        for row in df['date'].iteritems():
            dates.append(row[1][1:11])
        #----------------------------------------------------------------
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
                if str(article.title)=="Are you a robot?":
                    continue
                if str(article.text)=="":
                    continue
                article = {
                    "title": str(article.title),
                    "text": str(article.text),
                    "authors": article.authors,
                    "published_date": str(article.publish_date)[0:10],
                    "top_image": str(article.top_image),
                    "videos": article.movies,
                    "keywords": article.keywords,
                    "summary": str(article.summary)
                }
                #print("----------"+article["title"] + "------:" + article["text"] + "\n\n")
                articles.append(article);
            except: #if url can not be parsed by parser go to next entry
                continue

        return render_template("resultsBoth.html", articles=articles, tweets=tweets,cleanedTweets=cleanedTweets, cleanedSentiments=cleanedSentiments,tweetDates=dates)


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")


if __name__ == '__main__':
    app.run()
