"""
M5Forecast Evaluation
# Implementation of M5 Forecasting challenge on Kaggle, https://www.kaggle.com/c/m5-forecasting-accuracy/.

Created: 18 apr 2020
"""

# basic imports
import os
import pandas as pd

# own imports
from flow import load_data, select_dates, sales_to_money


class Referee(object):
    def __init__(self, sales_true, sales_train, prices, calendar):
        self.sales_true = sales_true
        self.h = sales_true.shape[1]
        self.n = sales_train.shape[1]

        # Calculate scale: 1/(n-1) * sum (Yt - Y_t-1) ^ 2
        self.scale = ((sales_train - sales_train.shift(1, axis=1)) ** 2).sum(axis=1) / (self.n - 1)

        # Calculate weights (cumulative dollars per product, in last 28 days of train data)
        # 1. Select final 28 days of training data
        last_train_day = int(sales_train.columns[-1][2:])  # remove 'd_' and convert to int
        sales_train_final = select_dates(sales_train, day_end=last_train_day, num_days=28)
        # 2. Convert quantities sold to money spent
        sales_train_final = sales_to_money(sales_train_final, prices, calendar, verbose=True)
        # 3. Calculate aggregate dollars spent
        self.weights = sales_train_final.sum(axis=1)

    def evaluate(self, sales_pred):
        """
        Evaluate the performance of the predicted sales. The predictions should provide an 28 day forecast for all
        3049 products of all 10 stores, so 30490x28 point forecasts.

        The evaluation metric is the WRMSSE: Weighed Root Mean Squared Scaled Error. Step by step:

        # 1. Calculate prediction error
        score = self.sales_true - sales_pred

        # 2. Calculate squared error
        score = score ** 2

        # 3. Sum over prediction horizon for each product
        score = score.sum(axis=1)

        # 4. Divide by scale
        score = score / self.scale

        # 5. Calculate mean (divide by number of predicted days)
        score = score / self.h

        # 6. Calculate root
        score = score ** (1/2)

        # 7. Multiply by weights
        score = score * self.weights

        :param sales_pred: 28 day point forecast of sales on product level.
        :return:
        """

        # Calculate WRMSSE
        score = ((((self.sales_true - sales_pred) ** 2).sum(axis=1) / self.scale / self.h) ** (1/2)) * self.weights

        return score


def test_referee():
    """Test the Referee evaluation"""
    # Load data
    calendar, sales_train_validation, sell_prices = load_data()

    # select true sales period
    sales_true = select_dates(sales_train_validation, day_end=1913, num_days=28)
    sales_train = select_dates(sales_train_validation, day_start=1, num_days=1913-28)

    # create referee with true sales
    ref = Referee(sales_true, sales_train, sell_prices, calendar)

    # create dummy predictions
    sales_pred = sales_true.copy()
    sales_pred = sales_pred * 0 + 1  # set all predictions to one

    # evaluate predictions
    score = ref.evaluate(sales_pred)
    print(score)


os.environ['DATA_DIR'] = 'data/'
test_referee()
