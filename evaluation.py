"""
M5Forecast Evaluation
# Implementation of M5 Forecasting challenge on Kaggle, https://www.kaggle.com/c/m5-forecasting-accuracy/.

Created: 18 apr 2020
"""

# basic imports
import os
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

# own imports
from flow import load_data, select_dates, sales_to_money, select_final_day


class Referee(object):
    def __init__(self, sales_true, sales_train, prices, calendar, verbose=True):
        self.sales_true = sales_true
        self.sales_train = sales_train
        self.h = sales_true.shape[1]
        self.n = sales_train.shape[1]

        # Define aggregation levels as their Pandas groupby
        # Follow same order as https://github.com/Mcompetitions/M5-methods/blob/master/validation/Point%20Forecasts%20-%20Benchmarks.R
        self.aggregation_levels = {
            1: lambda x: 1,  # global: 1
            2: ['state_id'],  # per state: 3
            3: ['store_id'],  # per store: 10
            4: ['cat_id'],  # per category: 3
            5: ['dept_id'],  # per department: 7
            6: ['state_id', 'cat_id'],  # per state & cat: 9
            7: ['state_id', 'dept_id'],  # per state & dep: 21
            8: ['store_id', 'cat_id'],  # per store & cat: 30
            9: ['store_id', 'dept_id'],  # per store & dep: 70
            10: ['item_id'],  # per item, across stores/states: 3049
            11: ['item_id', 'state_id'],  # per item, across stores: 9,225
            12: None  # lowest level, per product, per store
        }

        # Set number of aggregation levels
        self.K = len(self.aggregation_levels)  # 12 for full evaluation

        # Calculate weights (based on cumulative dollars per product, in last 28 days of train data)
        # Select final 28 days of training data
        last_train_day = select_final_day(sales_train)
        sales_train_final = select_dates(sales_train, day_end=last_train_day, num_days=28, include_metadata=True)

        # Convert quantities sold to money spent
        if verbose: print("Converting sales to money spent...")
        sales_train_final = sales_to_money(sales_train_final, prices, calendar, verbose=verbose)

        # Calculate weights of each level
        if verbose: print("Calculating weights for each level...")
        self.weights = {}
        for level, groupby in self.aggregation_levels.items():
            self.weights[level] = self.calc_weights(sales_train_final, groupby=groupby)

        # Calculate scale of each level
        if verbose: print("Calculating scale for each level...")
        self.scales = {}
        for level, groupby in self.aggregation_levels.items():
            self.scales[level] = self.calc_scale(groupby=groupby)

        if verbose: print("Finished setup.")

    def calc_weights(self, sales, groupby=None):
        """From the docs: Since the M5 competition involves twelve aggregation levels, K is set equal to 12,
        with the weights of the series being computed so that they sum to one at each aggregation level."""
        # Calculate aggregate dollars spent
        if groupby:
            sales = sales.groupby(groupby).sum()

        weights = sales.sum(axis=1)

        # Normalize weights to sum to one
        weights /= weights.sum()
        return weights

    def calc_scale(self, groupby=None):
        # Calculate scale: 1/(n-1) * sum (Yt - Y_t-1) ^ 2
        # agg level 3: 2.77 s ± 300 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # agg level 12: 3.91 s ± 921 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        if groupby:
            Yt = self.sales_train.groupby(groupby).sum()
            Yt1 =Yt.shift(1, axis=1)
        else:
            day_cols = self.sales_train.filter(regex='d_').columns
            Yt = self.sales_train[day_cols]
            Yt1 = Yt.shift(1, axis=1)

        scale = ((Yt - Yt1) ** 2).sum(axis=1) / (self.n - 1)
        return scale

    def evaluate(self, sales_pred):
        """
        Evaluate the performance of the predicted sales. The predictions should provide an 28 day forecast for all
        3049 products of all 10 stores, so 30490x28 point forecasts. With these forecasts, the WRMSSE for each
        level will be calculated by calculating the aggregated sales. The final evaluation score is the mean of
        these 12 WRMSSEs.
        """
        metrics = {}

        # Calculate WRMSSE for each level
        for level, groupby in self.aggregation_levels.items():
            # The groupby, weights and scale will be selected using the level
            metrics[level] = self.calc_WRMSSE(self.sales_true, sales_pred, level=level)

        WRMSSE = np.mean(list(metrics.values()))  # or sum and divide by self.K
        metrics['WRMSSE'] = WRMSSE
        return metrics

    def calc_WRMSSE(self, sales_true, sales_pred, level=None, groupby=None, weights=None, scale=None):
        """Calculate weighed root mean squared scaled error.
        Step by step:

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
        score = score * self.weights"""

        # Input validation
        if level:
            if groupby is None: groupby = self.aggregation_levels[level]
            if weights is None: weights = self.weights[level]
            if scale is None: scale = self.scales[level]
        assert weights is not None, "Provide level or weights"
        assert scale is not None, "Provide level or scale"

        # Calculate aggregate sales
        if groupby:
            pred = sales_pred.groupby(groupby).sum()
            true = sales_true.groupby(groupby).sum()
        else:
            day_cols = sales_true.filter(regex='d_').columns
            pred = sales_pred[day_cols]
            true = sales_true[day_cols]

        # Timeit of highest level WRMSSE: 24 ms ± 2.08 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
        # Timeit of lowest level WRMSSE: 20.8 ms ± 3.44 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
        score = (((((true - pred) ** 2).mean(axis=1)).T / scale).T ** (1 / 2))
        score = (score * weights.T).T.sum().sum()
        return score


def test_referee():
    """Test the Referee evaluation"""
    # Load data
    calendar, sales_train_validation, sell_prices = load_data()

    # select true sales period
    sales_true = select_dates(sales_train_validation, day_end=1913, num_days=28, include_metadata=True)
    sales_train = select_dates(sales_train_validation, day_start=1, num_days=1913-28, include_metadata=True)

    # create referee with true sales
    ref = Referee(sales_true, sales_train, sell_prices, calendar)

    # create dummy predictions
    sales_pred = sales_true.copy()
    day_cols = sales_pred.filter(regex='d_').columns
    sales_pred[day_cols] = sales_pred[day_cols] * 0 + 1  # set all predictions to one

    # set all predictions to training mean
    for dday in tqdm(day_cols, desc='calculating means'):
        sales_pred[dday] *= sales_train.filter(regex='d_').mean(axis=1)

    # evaluate predictions
    metrics = ref.evaluate(sales_pred)
    # Timeit of evaluate: 301 ms ± 38 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    print(metrics)


os.environ['DATA_DIR'] = 'data/'
test_referee()
