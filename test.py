from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
import warnings
from model import create_model
from itertools import product
from datetime import datetime
warnings.filterwarnings('ignore')
plt.style.use('seaborn-poster')

df=pd.read_csv('merged_data/merged_file.csv')

scaler = MinMaxScaler()

price_sentiment = df[['Price (INR)', 'Sentiment']].values
scaled_price_sentiment = scaler.fit_transform(price_sentiment)

seq_len = 60
dropout = 0.4
window_size = seq_len - 1

def split_into_sequences(data, seq_len):
    n_seq = len(data) - seq_len + 1
    return np.array([data[i:(i+seq_len)] for i in range(n_seq)])

def get_train_test_sets(data, seq_len, train_frac):
    sequences = split_into_sequences(data, seq_len)
    n_train = int(sequences.shape[0] * train_frac)
    x_train = sequences[:n_train, :-1, :]
    y_train = sequences[:n_train, -1, :]
    x_test = sequences[n_train:, :-1, :]
    y_test = sequences[n_train:, -1, :]
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = get_train_test_sets(scaled_price_sentiment, seq_len, train_frac=0.9)
# fraction of the input to drop; helps prevent overfitting

model = create_model(window_size, dropout, x_train)
checkpoint_filepath = 'results/best_model_weights.h5'
model.load_weights(checkpoint_filepath)
model.summary()


y_pred = model.predict(x_test)
c=y_test.copy()
c[:, 0] = y_pred[:, 0]
y_pred=c
# invert the scaler to get the absolute price data
y_test_orig = scaler.inverse_transform(y_test)
y_pred_orig = scaler.inverse_transform(y_pred)
plt.plot(y_test_orig[:,0], label='Actual Price', color='orange')
plt.plot(y_pred_orig[:,0], label='Predicted Price', color='green')

plt.title('BTC Price Prediction')
plt.xlabel('Hours')
plt.ylabel('Price (INR)')
plt.legend(loc='best')
plt.savefig('results/btc_price_prediction.png')