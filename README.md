# Bitcoin Price Prediction using Twitter Sentiment Analysis

## Overview
This repository contains code for predicting Bitcoin prices based on sentiment analysis of Twitter data.

-Preprocessing Twitter data from '/kaggle/input/bitcoin-tweets/Bitcoin_tweets.csv'.
-Performing sentiment analysis on the tweets using NLTK's VADER sentiment analyzer.
-Fetching Bitcoin's historical prices for the last 90 days based on recent tweets.
-Merging hourly sentiment data with Bitcoin historical prices.
-Implementing an LSTM-based deep learning model to predict Bitcoin prices based on sentiment and historical price data.

## Dependencies
- Python 3.x
- Libraries: pandas, nltk, matplotlib, tensorflow, scikit-learn, requests

## Setup
1. Download the Bitcoin tweets dataset from Kaggle and save it to `raw_data/Bitcoin_tweets.csv`.
2. Run `PreProcess.py` to preprocess Twitter data, perform sentiment analysis, and fetch historical Bitcoin prices.

## Usage
1. Run `PreProcess.py` to preprocess the data and save sentiment and historical price data in `cleaned_data/`.
2. Run `MergeData.py` to merge sentiment data with Bitcoin historical prices and save the resulting merged CSV in `merged_data/`.
3. Run `train.py` to create and train the LSTM-based model for Bitcoin price prediction. The best model weights will be saved in `results/`.
4. Run `test.py` to generate predictions and visualize Bitcoin price predictions. Results will be saved in `results/`.


## Folder Structure

```
- `PreProcess.py`: Preprocessing Twitter data, sentiment analysis, and fetching historical Bitcoin prices.
- `MergeData.py`: Merging sentiment data with Bitcoin historical prices.
- `model.py`: Contains functions to create and train the LSTM-based model for price prediction.
- `test.py`: Generates predictions and visualizes Bitcoin price predictions.
- `raw_data/`: Directory for raw Bitcoin tweets data.
- `cleaned_data/`: Directory for cleaned data output (sentiment and historical prices).
- `merged_data/`: Directory for merged data output.
- `results/`: Directory for model results and visualizations.
```

## Training and test
### Training
To train the model for epochs 10, batch_size 16, dropout 0.4 run:

```
python train.py
```
### Test
To test the model run:

```
python test.py
```


# Model 
| Model        | Parameters       | Trainable params |
| ------------ | ---------------- | ---------------- |
| LSTM         | 322495 (1.23 MB) | 322495 (1.23 MB) |
