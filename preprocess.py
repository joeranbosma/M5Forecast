"""
M5Forecast - Preprocess
# Implementation of M5 Forecasting challenge on Kaggle, https://www.kaggle.com/c/m5-forecasting-uncertainty/.

Created: 23 may 2020
"""

import os, gc
import numpy as np
import pandas as pd

from flow import restore_tags_converted_sales, read_converted_sales
from lightgbm_kernel import read_data, encode_categorical, reduce_mem_usage
from feature_extraction import aggregate_adapted_fe

from sklearn.preprocessing import LabelEncoder


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
def preprocess(level, n_years, DAYS_PRED=None, save_prepared_dataset=False, data_dir=None, augment_events=False):
    # read data for pipeline from lightgbm-poisson-w-scaled-pinball-loss.ipynb
    calendar, sell_prices, sales_train_val, submission = read_data()  # with memory reduction

    # Add days prior/after event as new event, if enabled
    if augment_events:
        calendar = augment_calendar_events(calendar)

    # read comverted sales
    converted_sales = read_converted_sales(level=level, data_dir=data_dir)

    ### Replace demand with normalised sales
    sales_train_val = converted_sales

    ## Count
    NUM_ITEMS = sales_train_val.shape[0]  # 1 / ... / 70 / ... / 30,240
    if DAYS_PRED is None:
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
        aug_fn_part = "_aug_events" if augment_events else ""

        fn = data_dir + 'prep/level_{}_simple_fe{}_{}_{}_normalised_demand_lag_{}.pickle'.format(
            level, aug_fn_part, data.date.min().date().strftime("%Y_%m_%d"), data.date.max().date().strftime("%Y_%m_%d"),
            DAYS_PRED
        )
        print("Saving to file..")
        data.to_pickle(fn)
        print("Finished.")

    return data


def reset_categorical_features(data, available_cat_features):
    # Set the NaNs in these categories as a single element
    for col in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
        data[col] = data[col].cat.add_categories(-1).fillna(-1)

    # re-index the categories to match the input of the Embedding layer
    for col in available_cat_features:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # re-set the categories as pandas categories for proper pd.get_dummies
    for col in available_cat_features:
        data[col] = data[col].astype('category')

    # set NaN values of shift_x, rolling_std_x, etc. to zero
    nan_cols = data.isna().sum()
    nan_cols = nan_cols[nan_cols != 0]

    for col in nan_cols.index:
        mask = data[col].isna()
        data.loc[mask, col] = 0

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


def get_features(level, prediction_lag=28, sell_price_features=True):
    features = [
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
        # demand features.
        "shift_t{}".format(prediction_lag),
        "shift_t{}".format(prediction_lag+1),
        "shift_t{}".format(prediction_lag+2),
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


def read_and_preprocess_data(level, data_dir='data/', day_start=None, prediction_lag=28,
                             augment_events=False, verbose=True):
    # try reading from file
    if day_start is None:
        day_start = '2014_04_26' if level == 12 else '2011_01_29'

    aug_fn_part = "_aug_events" if augment_events else ""

    day_end = '2016_04_24'
    fn = data_dir + 'prep/level_{}_simple_fe{}_{}_{}_normalised_demand_lag_{}.pickle'.format(
        level, aug_fn_part, day_start, day_end, prediction_lag)

    # check if already preprocessed
    if os.path.exists(fn):
        print("Reading from file {}..".format(fn)) if verbose else None
        data = pd.read_pickle(fn)
    else:
        # choose number of years to include
        n_years = 2 if level == 12 else 6

        # preform preprocessing
        data = preprocess(level=level, n_years=n_years, DAYS_PRED=prediction_lag, augment_events=augment_events,
                          save_prepared_dataset=True, data_dir=data_dir)

    print("Converting to pandas data..") if verbose else None
    # set categorical features
    data, available_cat_features = pandas_cat_data(data)

    print("Resetting categories..") if verbose else None
    # reset categorical features, set NaN events as additional category, set NaN shift/std/mean/kurt/skew to zero
    data = reset_categorical_features(data, available_cat_features)

    # select features
    features = get_features(level=level, prediction_lag=prediction_lag,
                            sell_price_features=('sell_price' in data.columns))
    print(features) if verbose else None

    return data, features, available_cat_features


def add_event(cal, idx, new_event_name, new_event_type, num=1, inplace=True):
    if not inplace:
        cal = cal.copy()

    # categorical calendar
    name_col = 'event_name_{}'.format(num)
    type_col = 'event_type_{}'.format(num)

    if str(cal[name_col].dtype) == "category":
        # create new categories
        if not new_event_name in cal[name_col].cat.categories:
            cal[name_col].cat.add_categories(new_event_name, inplace=True)
        if not new_event_type in cal[type_col].cat.categories:
            cal[type_col].cat.add_categories(new_event_type, inplace=True)

    # add 'new' event
    cal.loc[idx, name_col] = new_event_name
    cal.loc[idx, type_col] = new_event_type

    return cal


def augment_calendar_events(calendar, verbose=False):
    print("Adding days prior/after events to calendar..")

    cal = calendar.copy()
    d_days = [int(d[2:]) for d in list(cal.d.values)]
    d_start = min(d_days)
    d_end = max(d_days)

    for event_name in calendar.event_name_1.unique():
        if event_name is np.nan:
            continue

        # select event's days
        d_days = list(calendar[calendar.event_name_1 == event_name].d.values)
        print(event_name, d_days) if verbose else None

        # select event's type
        event_type = calendar[calendar.event_name_1 == event_name].event_type_1.iloc[0]

        # loop through week prior and week after
        for i in range(-7, 8):
            if i == 0:
                continue

            # create list of new d_days
            aug_d_nums = [int(d[2:]) + i for d in d_days]

            # loop over augmented list
            for d_num in aug_d_nums:
                if d_num < d_start or d_num > d_end:
                    continue
                d_day = 'd_' + str(d_num)

                # create new name for event
                postfix = ('+' if i > 0 else '') + str(i)
                new_event_name = event_name + postfix
                new_event_type = event_type + postfix

                # print(d_day, new_event_name)
                # check if date is 'available'
                idx = calendar[calendar.d == d_day].index[0]
                target_event = calendar.loc[idx, 'event_name_1']

                if target_event is np.nan:
                    add_event(cal, idx, new_event_name, new_event_type, num=1)
                elif calendar.loc[idx, 'event_name_2'] is np.nan:
                    add_event(cal, idx, new_event_name, new_event_type, num=2)
                else:
                    print("Skipped day {} to {}, both events filled".format(d_day, new_event_name)) if verbose else None

    return cal
