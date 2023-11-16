#Merge hourly sentiment of Bitcoin tweets with Bitcoin historical prices based on timestamps.
import pandas as pd

df1 = pd.read_csv('cleaned_data/Sentiment.csv')
df2 = pd.read_csv('cleaned_data/bitcoin_historical_prices.csv')


merged_df = pd.merge(df1, df2, on='Datetime', how='inner')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_data/merged_file.csv', index=False)