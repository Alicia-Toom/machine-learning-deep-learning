from models.prediction import Prediction
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import datetime as dt


class PredictionService:
    def __init__(self, ticker):
        self._ticker = ticker
        self._model = None
        pass

    def train(self):
        # Läsa data
        Df = yf.download(self._ticker, '2011-01-01', '2021-7-18', auto_adjust=True)

        # Håll bara nära kolumner
        Df = Df[['Close']]

        # Släpp rader med saknade värden
        Df = Df.dropna()

        # Define explanatory variables
        Df['S_3'] = Df['Close'].rolling(window=3).mean()
        Df['S_9'] = Df['Close'].rolling(window=9).mean()
        Df['next_day_price'] = Df['Close'].shift(-1)

        Df = Df.dropna()
        X = Df[['S_3', 'S_9']]

        # Define dependent variable
        y = Df['next_day_price']

        # Split the data into train and test dataset
        t = .8
        t = int(t * len(Df))

        # Train dataset
        X_train = X[:t]
        y_train = y[:t]

        # Test dataset
        X_test = X[t:]
        y_test = y[t:]

        # Create a linear regression model
        linear = LinearRegression().fit(X_train, y_train)
        print("Linear Regression model")
        print("TESLA ETF Price (y) = %.2f * 3 Days Moving Average (x1) \
        + %.2f * 9 Days Moving Average (x2) \
        + %.2f (constant)" % (linear.coef_[0], linear.coef_[1], linear.intercept_))

        # Predicting the Gold ETF prices
        predicted_price = linear.predict(X_test)
        predicted_price = pd.DataFrame(
            predicted_price, index=y_test.index, columns=['price'])
        # R square
        r2_score = linear.score(X[t:], y[t:]) * 100
        float("{0:.2f}".format(r2_score))

        tesla = pd.DataFrame()

        tesla['price'] = Df[t:]['Close']
        tesla['predicted_price_next_day'] = predicted_price
        tesla['actual_price_next_day'] = y_test
        tesla['tesla_returns'] = tesla['price'].pct_change().shift(-1)

        tesla['signal'] = np.where(tesla.predicted_price_next_day.shift(1) < tesla.predicted_price_next_day, 1, 0)

        tesla['strategy_returns'] = tesla.signal * tesla['tesla_returns']

        # Calculate sharpe ratio
        sharpe = tesla['strategy_returns'].mean() / tesla['strategy_returns'].std() * (252 ** 0.5)
        'Sharpe Ratio %.2f' % (sharpe)


    def predict(self):
        #current_date = dt.datetime.now()

        # Get the data
        #data = yf.download(self._ticker, '2011-06-01', current_date, auto_adjust=True)
        #data['S_3'] = data['Close'].rolling(window=3).mean()
        #data['S_9'] = data['Close'].rolling(window=9).mean()
        #data = data.dropna()

        # Forecast the price
        #data['predicted_tesla_price'] = linear.predict(data[['S_3', 'S_9']])
        #data['signal'] = np.where(data.predicted_tesla_price.shift(1) < data.predicted_tesla_price, "Buy", "No Position")

        # Print the forecast
        #data.tail(1)[['signal', 'predicted_tesla_price']].T  # import datetime and get today's date
        return Prediction(self._ticker)


    def plot(self):
        pass