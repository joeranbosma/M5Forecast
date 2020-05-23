"""
M5Forecast - Preprocess
# Implementation of M5 Forecasting challenge on Kaggle, https://www.kaggle.com/c/m5-forecasting-uncertainty/.

Created: 23 may 2020
"""

import gc
import numpy as np
import pandas as pd

from flow import restore_tags_converted_sales, read_converted_sales
from lightgbm_kernel import read_data, encode_categorical, reduce_mem_usage
from feature_extraction import aggregate_adapted_fe


categorical_features = {
    1: [],
    2: ['state_id'],
    3: ['state_id', 'store_id'],
    4: ['cat_id'],
    5: ['cat_id', 'dept_id'],
    6: ['state_id', 'cat_id'],
    7: ['state_id', 'cat_id', 'dept_id'],
    8: ['state_id', 'store_id', 'dept_id'],
    9: ['state_id', 'store_id', 'cat_id', 'dept_id'],
    10: ['cat_id', 'dept_id', 'item_id'],
    11: ['state_id', 'cat_id', 'dept_id', 'item_id'],
    12: ['state_id', 'store_id', 'cat_id', 'dept_id', 'item_id'],
}


# adapted from from https://www.kaggle.com/robertburbidge/lightgbm-poisson-w-scaled-pinball-loss
def preprocess(level, n_years, save_prepared_dataset=False, data_dir=None):
    # read data for pipeline from lightgbm-poisson-w-scaled-pinball-loss.ipynb
    calendar, sell_prices, sales_train_val, submission = read_data()  # with memory reduction

    # read comverted sales
    converted_sales = read_converted_sales(level=level, data_dir=data_dir)

    ### Replace demand with normalised sales
    sales_train_val = converted_sales

    ## Count
    NUM_ITEMS = sales_train_val.shape[0]  # 1 / ... / 70 / ... / 30,240
    DAYS_PRED = submission.shape[1] - 1  # 28
    print(NUM_ITEMS, DAYS_PRED)

    nrows = int(365 * n_years * NUM_ITEMS)

    ## Encode categorical features
    calendar = encode_categorical(
        calendar, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    ).pipe(reduce_mem_usage)

    sales_train_val = encode_categorical(
        sales_train_val, categorical_features[level],
    ).pipe(reduce_mem_usage)

    sell_prices = encode_categorical(sell_prices, ["item_id", "store_id"]).pipe(
        reduce_mem_usage
    )

    ## Reshape
    sales_train_val = pd.melt(sales_train_val,
                              id_vars=['id', *categorical_features[level]],
                              var_name='day', value_name='demand')
    print('Melted sales train validation has {} rows and {} columns'.format(*sales_train_val.shape))
    sales_train_val.head()

    print("Selecting {} rows ({:.1%})".format(nrows, nrows / sales_train_val.index.size))
    data = sales_train_val.iloc[-nrows:, :]

    ## Add calendar features
    # calendarの結合
    # drop some calendar features(不要な変数の削除:weekdayやwdayなどはdatetime変数から後ほど作成できる。)
    calendar.drop(['weekday', 'wday', 'month', 'year'],
                  inplace=True, axis=1)

    # notebook crash with the entire dataset (maybe use tensorflow, dask, pyspark xD)(dayとdをキーにdataに結合)
    data = pd.merge(data, calendar, how='left', left_on=['day'], right_on=['d'])
    data.drop(['d', 'day'], inplace=True, axis=1)

    # add sell price if all of the columns 'store_id', 'item_id', 'wm_yr_wk' are available
    # sell price
    if np.prod([col in data.columns for col in ['store_id', 'item_id', 'wm_yr_wk']]):
        # get the sell price data (this feature should be very important)
        data = data.merge(sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))

    # memory
    del sell_prices
    gc.collect()

    data = reduce_mem_usage(aggregate_adapted_fe(data, DAYS_PRED=DAYS_PRED))

    if save_prepared_dataset:
        fn = data_dir + 'prep/level_{}_simple_fe_{}_{}_normalised_demand.pickle'.format(
            level, data.date.min().date().strftime("%Y_%m_%d"), data.date.max().date().strftime("%Y_%m_%d")
        )
        print("Saving to file..")
        data.to_pickle(fn)
        print("Finished.")

    return data


def pandas_cat_data(data):
    cat_features = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        # time features.
        "year",
        "month",
        "week",
        "day",
        "dayofweek",
    ]

    available_cat_features = [f for f in cat_features if f in data.columns]

    for col in available_cat_features:
        data[col] = data[col].astype('category')

    return data, available_cat_features


def get_features(level, sell_price_features=True):
    features = [
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
        # demand features.
        "shift_t28",
        "shift_t29",
        "shift_t30",
        "rolling_std_t7",
        "rolling_std_t30",
        "rolling_std_t60",
        "rolling_std_t90",
        "rolling_std_t180",
        "rolling_mean_t7",
        "rolling_mean_t30",
        "rolling_mean_t60",
        "rolling_mean_t90",
        "rolling_mean_t180",
        "rolling_skew_t30",
        "rolling_kurt_t30",
        # time features.
        "year",
        "month",
        "week",
        "day",
        "dayofweek",
        "is_year_end",
        "is_year_start",
        "is_quarter_end",
        "is_quarter_start",
        "is_month_end",
        "is_month_start",
        "is_weekend"
    ]

    # add categorical features, based on availability
    features += categorical_features[level]

    # add price features, based on availability
    if sell_price_features:
        features += [
            # price features
            "sell_price",
            "price_change_t1",
            "price_change_t365",
            "rolling_price_std_t7",
            "rolling_price_std_t30",
        ]

    return features
