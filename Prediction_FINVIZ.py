## Sentiment Analysis with news from FinViz
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

def print_Headlines(coy_tr):
    for index, table_row in enumerate(coy_tr):
        headlines = table_row.a.text
        datetime = table_row.td.text
        print(headlines)
        print(datetime)
        
        # Print out the 4 latest news articles
        if index == 4:
            break

news_table = html.find(id='news-table')
coy_tr = news_table.findAll('tr')
# news_tables[ticker] = news_table
# company = news_tables[ticker]









