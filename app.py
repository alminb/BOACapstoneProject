import os
import pandas as pd
from datetime import date
from flask import Flask,request,render_template,session,redirect,url_for,flash

app = Flask(__name__)


@app.route('/')
def entry():
    return render_template("home.html")

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/results',methods=['POST','GET'])
def results():
    if request.method == 'POST':
        start_day = date.today()
        end_date = start_day
        print(end_date)
        twitterHashtag = request.form['twitterHashtag']
        search_term = "#"+twitterHashtag
        from_date = "2022-03-22"
        os.system(f"snscrape --since {from_date} twitter-search '{search_term}' > result-tweets.txt")
        if os.stat("result-tweets.txt").st_size == 0:
            counter = 0
        else:
            df = pd.read_csv('result-tweets.txt',on_bad_lines='skip',names=['link'])
            counter = df.size
        print('Number of Tweets : ' + str(counter))
        max_results = 100
        extracted_tweets = "snscrape --format '{content!r}'" + f" --max-results {max_results} --since {from_date} twitter-search '{search_term}' > extracted-tweets.txt"
        os.system(extracted_tweets)
        if os.stat("extracted-tweets.txt").st_size == 0:
            print("No tweets found")
        else:
            df = pd.read_csv('extracted-tweets.txt',delimiter="''",skipinitialspace=True, names=['content'])
            for row in df['content'].iteritems():
                print(row)

        return render_template("results.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")


if __name__ == '__main__':
    app.run()
