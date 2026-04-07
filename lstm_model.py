import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_lstm(data):
    return data  # no heavy model

def predict_traffic(data, last_values):
    # simple average prediction
    return int(sum(last_values) / len(last_values))

    X = []
    y = []

    for i in range(len(data)-3):
        X.append(data[i:i+3])
        y.append(data[i+3])

    X = np.array(X)
    y = np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(3,1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, verbose=0)

    return model

def predict_traffic(model, last_values):
    last_values = np.array(last_values).reshape(1,3,1)
    return int(model.predict(last_values)[0][0])