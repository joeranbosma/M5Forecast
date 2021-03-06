"""
M5Forecast - Agent
# Implementation of M5 Forecasting challenge on Kaggle, https://www.kaggle.com/c/m5-forecasting-accuracy/.
Here an Agent is created to make predicting the sales behave more like scikit-learn models.

Each model should be able to be trained using a .fit() method,
with relevant hyperparameters set at initialization.
Predictions of the 28 day period should possible with the .predict() function.

Created: 23 apr 2020
"""

# basic imports
import numpy as np
import pandas as pd

# own imports
from flow import select_dates, select_final_day, select_day_nums, get_empty_predictions


class AgentBase(object):
    """Base class for future implementation"""
    def __init__(self, verbose=True):
        pass


class KDayMeanTimesWeeklyPattern(AgentBase):
    """Calculates the mean of the final k training days and multiplies those with the typical weekly pattern.
     This class uses a simple average and does not calculate a rolling mean. """
    def __init__(self, calendar, k=28):
        super().__init__()
        self.calendar = calendar
        self.k = k
        self.portions = None

    def fit(self, sales_train):
        """Calculate the weekly pattern"""
        col_list = []

        # Gather which columns/days correspond to which day of the week
        for i in range(1, 1 + 7):
            # select days from a certain day of the week
            cols = list(self.calendar[self.calendar.wday == i].d.values)
            # filter days to match training set
            cols = [d for d in cols if d in sales_train.columns]
            col_list.append(cols)

        # Calculate total number of sales per day of the week
        num_sales = [sales_train[cols].sum(axis=1).sum() for cols in col_list]

        # Calculate portion of total
        portions = np.array(num_sales) / np.sum(num_sales)
        self.portions = portions

    def predict(self, sales_train, day_start=None, day_end=None):
        """Predict the next 28 days based on the mean times the weekly pattern"""
        if day_start is None:
            # infer start day as first day after training set
            day_start = select_final_day(sales_train) + 1
        if day_end is None:
            # infer end day as start day + 28 day period
            day_end = day_start + 28 - 1  # inclusive range

        # get skeleton for predictions
        pred = get_empty_predictions()

        # determine mean of last k days, per product
        df = select_dates(sales_train, num_days=self.k, day_end=select_final_day(sales_train))
        d_cols = select_day_nums(df, as_int=False)
        weekly_mean = df[d_cols].mean(axis=1) * 7

        for day_num in range(day_start, day_end+1):
            d_day = 'd_%d' % day_num
            week_day = self.calendar[self.calendar.d == d_day].wday.values[0]
            portion = self.portions[week_day - 1]
            # set all predictions to training mean of last k days times weekly pattern
            pred[d_day] = weekly_mean * portion

        return pred


class AggregateAgent(AgentBase):
    """Agent to predict sales on aggregate level"""
    # train_norm: float
    # features: list or dict
    # labels: list
    # window_in: int

    def __init__(self, model, train_norm, features, labels, window_in=28):
        super().__init__()
        self.model = model
        self.train_norm = train_norm
        self.features = features
        self.labels = labels
        self.window_in = window_in

    def predict(self, train_df, val_day_nums):
        # select date columns
        final_inp_day = min(val_day_nums) - 1
        first_inp_day = final_inp_day - self.window_in + 1
        inp_day_idx = ['d_%d' % d for d in range(first_inp_day, final_inp_day + 1)]
        out_day_idx = ['d_%d' % d for d in val_day_nums]

        # predict (wrap and unwrap into 'batch' of one)
        if isinstance(self.features, dict):
            X = {}
            for key, feats in self.features.items():
                X[key] = np.array([train_df.loc[inp_day_idx, feats].values])
        else:
            X = np.array([train_df.loc[inp_day_idx, self.features].values])
        y_pred = self.model.predict(X)[0]

        # reshape, set columns and indices, and de-normalise
        sales_pred = pd.DataFrame(y_pred * self.train_norm,
                                  index=out_day_idx, columns=self.labels)
        return sales_pred
