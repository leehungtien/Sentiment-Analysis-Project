## Sentiment Analysis with news from FinViz
from numpy.core.fromnumeric import mean
import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
import matplotlib.pyplot as plt
# NLTK VADER for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

finviz_url = 'https://finviz.com/quote.ashx?t='

# news_tables = {}

# Get company from user
ticker = input("Please enter company's ticker symbol: ")

url = finviz_url + ticker 
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36',
}
response = requests.get(url, headers=HEADERS)
html = BeautifulSoup(response.text, features="html.parser")

def loadNews(ticker, coy_tr, news):
    for index, table_row in enumerate(coy_tr):

        # Headlines are found html <a> tag
        headlines = table_row.a.text

        # Datetime are found under html <td> tag
        datetime = table_row.td.text
        date_scrape = datetime.split()

        # If date_scrape is of length 1, it only has the information regarding time.
        if (len(date_scrape) == 1):
            time = date_scrape[0]

        elif (len(date_scrape) == 2):
            date = date_scrape[0]
            time = date_scrape[1]

        news.append([date, time, headlines])
       
        # Print out the last 4 news articles that were published
        if index < 4:
            print(headlines)
            print(datetime)
        
        

# News are found under the id: news-table
news_table = html.find(id='news-table')
# news_tables[ticker] = news_table
# company = news_tables[ticker]
coy_tr = news_table.findAll('tr')

parsed_news = []
loadNews(ticker, coy_tr, parsed_news)

vader = SentimentIntensityAnalyzer()
columns = ['Date', 'Time', 'Headline']

# Convert parsed_news into a Pandas Dataframe
df = pd.DataFrame(parsed_news, columns=columns)
scores = df['Headline'].apply(vader.polarity_scores).tolist()
scores_df = pd.DataFrame(scores)
newsAndScores_df = df.join(scores_df)

# Group data by dates and calculate the mean score of each day
mean_scores = newsAndScores_df.groupby(['Date']).mean()

mean_scores = mean_scores.xs('compound', axis='columns').transpose()

mean_scores.plot(kind='bar')
plt.grid()
plt.show()









