from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.keras.models import Sequential

def create_model(window_size, dropout, x_train):
    model = Sequential()
    model.add(LSTM(window_size, return_sequences=True,input_shape=(window_size, x_train.shape[-1])))
    model.add(Dropout(rate=dropout))
    model.add(Bidirectional(LSTM((window_size * 2), return_sequences=True)))
    model.add(Dropout(rate=dropout))
    model.add(Bidirectional(LSTM(window_size, return_sequences=False)))
    model.add(Dense(units=1))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model