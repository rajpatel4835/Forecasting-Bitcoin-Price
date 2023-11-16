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
from itertools import product
from datetime import datetime
from model import create_model
warnings.filterwarnings('ignore')
plt.style.use('seaborn-poster')

df=pd.read_csv('merged_data/merged_file.csv')

scaler = MinMaxScaler()

price_sentiment = df[['Price (INR)', 'Sentiment']].values
scaled_price_sentiment = scaler.fit_transform(price_sentiment)

# Divide the data into shorter-period sequences
seq_len = 60
batch_size = 16
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

# Build a 3-layer LSTM RNN
model = create_model(window_size, dropout, x_train)


# Define the ModelCheckpoint callback
checkpoint_filepath = 'results/best_model_weights.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=batch_size,
    shuffle=False,
    validation_split=0.2,
    callbacks=[model_checkpoint_callback]
)
model.load_weights(checkpoint_filepath)
model.summary()
