import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import matplotlib.pyplot as plt
import nltk
import csv
import requests
from datetime import datetime, timedelta
from dateutil.parser import parse


nltk.download('vader_lexicon')

plt.style.use('fivethirtyeight')

chunk = pd.read_csv('raw_data/Bitcoin_tweets.csv',chunksize=100000,lineterminator='\n')
f = pd.concat(chunk)


#removing duplicates
f.drop_duplicates(inplace = True)
f.reset_index(drop=True,inplace=True)
print("Shape after removing duplicates :",f.shape)

# Convert 'date' column to datetime
f['date'] = pd.to_datetime(f['date'], errors='coerce')

# Drop rows with invalid datetime values
f = f.dropna(axis=0 ,subset=['date','text'])
f.reset_index(drop= True,inplace=True)
print("Shape after droping invalid datetime value :",f.shape)


# Text Cleaning
def clean_tweet(twt):
    twt = re.sub('#bitcoin', 'bitcoin', twt, flags=re.IGNORECASE)
    twt = re.sub('#Bitcoin', 'bitcoin', twt, flags=re.IGNORECASE)
    twt = re.sub('#btc', 'bitcoin', twt, flags=re.IGNORECASE)
    twt = re.sub('#[A-Za-z0-9]+', '', twt)
    twt = re.sub(r'\n', '', twt)
    twt = re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', twt, flags=re.MULTILINE)
    twt = re.sub('@\\w+ *', '', twt, flags=re.MULTILINE)
    return twt

f['CleanTwt'] = f['text'].apply(clean_tweet)

# Sentiment Analysis using VADER
sia = SentimentIntensityAnalyzer()
f['Sentiment'] = f['CleanTwt'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Create a new DataFrame with only 'date' and 'Sentiment'
hourly_sentiment = f[['date', 'Sentiment']].copy()

# Group by hour and calculate mean sentiment scores
hourly_sentiment['Datetime'] = hourly_sentiment['date'].dt.floor('h')
hourly_sentiment = hourly_sentiment.groupby('Datetime')['Sentiment'].mean().reset_index()

# Create a complete date range from the minimum to maximum date
full_date_range = pd.date_range(hourly_sentiment['Datetime'].min(), hourly_sentiment['Datetime'].max(), freq='H')

# Merge with the complete date range to fill missing hours
hourly_sentiment_complete = pd.DataFrame({'Datetime': full_date_range})
hourly_sentiment_complete = pd.merge(hourly_sentiment_complete, hourly_sentiment, on='Datetime', how='left')

# Use linear interpolation to fill missing values
hourly_sentiment_complete['Sentiment'] = hourly_sentiment_complete['Sentiment'].interpolate(method='linear')

# Save the results to a CSV file
hourly_sentiment_complete.to_csv('cleaned_data/Sentiment.csv', index=False)
print("Bitcoin Tweet Sentiment saved to CSV successfully.")


# Filter the data for the last 90 days
end_date = hourly_sentiment_complete['Datetime'].max()+timedelta(hours=1)  # Get the latest date in the dataset
start_date = end_date - timedelta(days=90)

#Bitcoin historical prices for the past 90 days based on Bitcoin Tweet Sentiment.
def get_bitcoin_historical_prices(start_date, end_date):
  url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range'
  params = {
      'vs_currency': 'inr',
      'from': int(start_date.timestamp()),
      'to': int(end_date.timestamp()),
      # 'interval': 'hourly',
  }

  response = requests.get(url, params=params)

  if response.status_code == 200:
    data = response.json()
    prices = data.get('prices', [])
    return prices
  else:
    print(f"Error: {response.status_code}")
    print(response.text)
    return None

def save_bitcoin_historical_prices_to_csv(prices, filename):
  with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Datetime', 'Price (INR)'])

    for timestamp, price in prices:
      formatted_time = parse(datetime.utcfromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')).replace(minute=0, second=0, microsecond=0)
      writer.writerow([formatted_time, price])

if __name__ == "__main__":

  prices = get_bitcoin_historical_prices(start_date, end_date)

  if prices is not None:
    save_bitcoin_historical_prices_to_csv(prices, 'cleaned_data/bitcoin_historical_prices.csv')
    print("Bitcoin historical prices saved to CSV successfully.")
  else:
    print("Failed to fetch Bitcoin historical prices data.")