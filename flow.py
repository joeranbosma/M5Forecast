"""
M5Forecast - Data flow
# Implementation of M5 Forecasting challenge on Kaggle, https://www.kaggle.com/c/m5-forecasting-accuracy/.

Created: 18 apr 2020
"""

import os
import pandas as pd
from tqdm import tqdm as tqdm
import time


def load_data(data_dir = None):
    if data_dir is None:
        data_dir = os.environ['DATA_DIR']

    calendar = pd.read_csv(data_dir + 'calendar.csv')
    sales_train_validation = pd.read_csv(data_dir + 'sales_train_validation.csv')
    sell_prices = pd.read_csv(data_dir + 'sell_prices.csv')

    # convert columns to correct dtypes
    # calendar: convert categorical variables
    for col in ['weekday', 'd', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
        calendar[col] = calendar[col].astype('category')
    # calendar: set index to date
    calendar.index = pd.DatetimeIndex(calendar.date)
    calendar.drop(columns=['date'], inplace=True)

    # sales: convert categorical variables
    for col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']:
        sales_train_validation[col] = sales_train_validation[col].astype('category')
    # sales: set id as index
    sales_train_validation.set_index('id', inplace=True)

    # sell prices: convert categorical variables
    for col in ['store_id', 'item_id']:
        sell_prices[col] = sell_prices[col].astype('category')

    return calendar, sales_train_validation, sell_prices


def select_dates(df, day_start=None, num_days=None, day_end=None, include_metadata=False):
    """Select dates of the form d_1910 in the selected period (inclusive)"""

    if day_end is None:
        assert num_days is not None, "Set two of day_start, num_days and day_end"
        day_end = day_start + (num_days - 1)

    if day_start is None:
        assert num_days is not None, "Set two of day_start, num_days and day_end"
        day_start = day_end - (num_days - 1)

    # create list of days to select
    d_list = ["d_%d"%i for i in range(day_start, day_end+1)]

    # include metadata
    if include_metadata:
        d_list.extend(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])

    # select days from df and return
    return df[d_list]


def select_day_nums(df, as_int=True):
    # Select columns with 'd_'
    d_list = [col for col in df.columns if 'd_' in col]
    # remove 'd_' and convert to int
    if as_int:
        d_list = [int(col[2:]) for col in d_list]
    return d_list


def select_final_day(df):
    """Select final day of DataFrame"""
    d_list = select_day_nums(df)
    return max(d_list)


def sales_to_money(sales, prices, calendar, verbose=False):
    """Convert the quantities sold to dollars spent.
    Prices are provided on a weekly basis.
    """
    sales = sales.copy()

    d_list = list(sales.filter(regex='d_').columns)
    d_done = []
    iterator = d_list
    if verbose:
        iterator = tqdm(iterator, desc='Sales to money spent')

    for dday in iterator:
        if dday in d_done:
            continue

        # Fetch product prices
        # 1. Select week of dday
        wm_yr_wk = calendar[calendar.d == dday].wm_yr_wk.values[0]

        # 2. Select days in that week AND in the sales period
        week_days = calendar[calendar.wm_yr_wk == wm_yr_wk].d.values
        week_days = [d for d in week_days if d in d_list]

        # 3. Select prices of that week
        week_prices = prices[prices.wm_yr_wk == wm_yr_wk]

        # 4. Construct ids of products with price
        # sales have id like 'HOBBIES_1_001_CA_1_validation', prices have 'store_id' and 'item_id'
        indx = week_prices.apply(lambda x: x.item_id + '_' + x.store_id + '_validation', axis=1)

        # 5. Convert quantities sold to dollars spent
        sales.loc[indx, week_days] = (sales.loc[indx, week_days].T * week_prices.sell_price.values).T

        # 6. Track finished days
        d_done.extend(list(week_days))

    return sales


def create_submission(sales_pred, submission_dir=None, filename=None, add_timestamp=False):
    """Create submission file in csv format
    of sales predictions"""

    if submission_dir is None:
        submission_dir = os.environ['SUB_DIR']

    # Drop meta-columns and rename columns to F1, ..., F28
    pred_val = sales_pred.drop(columns=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
    pred_val.columns = ['F%d' % d for d in range(1, 28 + 1)]

    # At this point the second 28-day period is just the same
    pred_eval = pred_val.copy()
    pred_eval.index = [d.replace('_validation', '_evaluation') for d in pred_eval.index]

    sub = pd.concat((pred_val, pred_eval))
    sub.index.name = 'id'

    if filename is not None:
        timestamp = ''
        if add_timestamp:
            timestamp = time.strftime('_%Y-%m-%d_%H%M', time.localtime())
        sub.to_csv(submission_dir + "submission_{}{}.csv".format(filename,timestamp))
    else:
        timestamp = time.strftime('%Y-%m-%d_%H%M', time.localtime())
        sub.to_csv(submission_dir + "submission_{}.csv".format(timestamp))

