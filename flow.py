"""
M5Forecast - Data flow
# Implementation of M5 Forecasting challenge on Kaggle, https://www.kaggle.com/c/m5-forecasting-accuracy/.

Created: 18 apr 2020
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
import time
import matplotlib.pyplot as plt
from lightgbm_kernel import reduce_mem_usage
import pickle


def save_object(obj, fn):
    with open(fn, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(fn):
    with open(fn, 'rb') as handle:
        obj= pickle.load(handle)
    return obj


def load_data(data_dir=None):
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


def get_empty_predictions(data_dir=None, with_eval_cols=False):
    if data_dir is None:
        data_dir = os.environ['DATA_DIR']

    if with_eval_cols:
        empty_pred = pd.read_csv(data_dir + '/empty_pred.csv', index_col='id')
    else:
        empty_pred = pd.read_csv(data_dir + '/pred_skeleton.csv', index_col='id')
    return empty_pred


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
        meta_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        d_list.extend([col for col in meta_cols if col in df.columns])

    # select days from df and return
    return df[d_list]


def select_day_nums(df, as_int=True, axis=1):
    # Select columns with 'd_'
    cols = df.columns if axis == 1 else df.index
    d_list = [col for col in cols if 'd_' in col]
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


def create_uncertainty_submission_from_ref_format(sales_pred, submission_dir=None, filename=None, add_timestamp=False):
    """Create submission file in csv format, from df in format accepted by Referee"""

    if submission_dir is None:
        submission_dir = os.environ['SUB_DIR']

    # check if id column is in correct format
    assert 'Total_X_0.005_evaluation' in sales_pred['id'].values, "Supply id column in submission format"

    # Drop meta-columns and rename columns to F1, ..., F28
    preds1 = sales_pred.drop(columns=['quantile', 'level'])
    preds1 = preds1.set_index('id')
    preds1.columns = ['F%d' % d for d in range(1, 28 + 1)]

    # At this point the second 28-day period is just the same
    if '_evaluation' in preds1.index[0]:
        needle = '_evaluation'
        replacement = '_validation'
    else:
        needle = '_validation'
        replacement = '_evaluation'
    preds2 = preds1.copy()
    preds2.index = [d.replace(needle, replacement) for d in preds2.index]

    sub = pd.concat((preds1, preds2))
    sub.index.name = 'id'

    if filename is not None:
        timestamp = ''
        if add_timestamp:
            timestamp = time.strftime('_%Y-%m-%d_%H%M', time.localtime())
        filename = submission_dir + "submission_{}{}".format(filename, timestamp)
    else:
        timestamp = time.strftime('%Y-%m-%d_%H%M', time.localtime())
        filename = submission_dir + "submission_{}".format(timestamp)

    filename += ".csv.gz"
    sub.to_csv(filename, float_format='%.3f', compression='gzip')

    return filename


def model_predict(model, val_batch_creator, verbose=True):
    if verbose: print("Predicting...")
    # predict
    y_pred = model.predict(val_batch_creator)

    # match prediction with id's and stuff
    df = val_batch_creator.df[['id', 'date', 'demand']].copy()
    for i in range(len(y_pred)):
        df['pred_q{}'.format(i)] = y_pred[i].squeeze()
    return df


def denorm_preds(df, data_dir, level=9, verbose=True):
    if verbose: print("Denormalising...")
    norm = pd.read_csv(data_dir + 'prep/norm_level_{}.csv'.format(level), index_col='id')
    df = df.copy()
    pred_cols = [col for col in df.columns if ('pred_' in col or 'demand' in col)]
    df[pred_cols] = df.apply(lambda row: row[pred_cols] * norm.loc[row.id].norm, axis=1)
    return df


def warp_preds_to_ref_form(df, calendar, quantiles, level, verbose=True):
    if verbose: print("Warping predictions...")
    # Intitialize magic warp
    pred_cols = ['pred_q%d' % d for d in range(9)]
    df = df.melt(id_vars=['id', 'date'], value_vars=pred_cols, var_name="quantile", value_name="prediction")

    # map 'pred_q0' --> '0.005', etc.
    quantile_map = {'pred_q%d' % d: "{:.3f}".format(q) for (d, q) in enumerate(quantiles)}

    # Prepare magic
    df['id_q'] = df['id'] + '|' + df['quantile']
    df = df.pivot(index='id_q', columns='date', values='prediction')
    df = df.reset_index()

    # Perform magic
    df['id'] = df.apply(lambda row: row.id_q.split('|')[0], axis=1)
    df['quantile'] = df.apply(lambda row: quantile_map[row.id_q.split('|')[1]], axis=1)
    df = df.drop(columns=['id_q'])
    df['level'] = level
    df['id'] = df['id'] + '_' + df['quantile'] + '_evaluation'
    df['quantile'] = df['quantile'].astype(float)

    # Finalise magic
    if 'date' in calendar.columns:
        calendar = calendar.set_index('date')
        calendar.index = pd.to_datetime(calendar.index)
    cols = [calendar.loc[col].d if isinstance(col, pd.Timestamp) else col for col in df.columns]
    df.columns = cols
    d_cols = select_day_nums(df, as_int=False)
    df[d_cols] = df[d_cols].astype(float)
    return df


def plot_some(pred_df, ref, level, q=0.500):
    d_cols = select_day_nums(pred_df, as_int=False)
    # select true sales
    real_sales = ref.sales_true_quantiles.loc[
        (ref.sales_true_quantiles.level == level) & (ref.sales_true_quantiles['quantile'] == q),
        d_cols]

    # select predicted sales
    df = pred_df[pred_df['quantile'] == q]

    # plot
    f, axes = plt.subplots(2, 3, figsize=(18, 12))

    # plot the first 6 series, if available
    for i, ax in zip(range(df.index.size), np.ravel(axes)):
        real_sales.iloc[i].T.plot(ax=ax, label="True")
        df.iloc[i][d_cols].plot(ax=ax, label="Pred")
        ax.legend()
        ax.set_title(df.iloc[i].id)
        ax.set_ylabel("Sales")


def evaluate_model(model, ref, val_batch_creator, calendar, quantiles, data_dir, level, validation_set=False, verbose=True):
    # calculate model predictions
    df = model_predict(model, val_batch_creator)

    # denormalise model predictions
    df = denorm_preds(df, data_dir=data_dir, level=level)

    # perform absolute magic
    df = warp_preds_to_ref_form(df, calendar=calendar, quantiles=quantiles, level=level)

    if validation_set:
        return df

    if verbose: print("Evaluating..")
    # calculate (and display) WSPL
    metrics = ref.evaluate_WSPL(df)
    if verbose: print(metrics)

    if verbose:
        # preview some predictions
        plot_some(df, ref, level=level)

    return metrics, df


def restore_tags_converted_sales(df, level):
    if level == 1:
        # completely aggregated
        pass
    if level in [2, 3, 6, 7, 8, 9, 11]:
        # restore state id from start
        df['state_id'] = df.apply(lambda row: row.id.split('_')[0], axis=1)
    if level in [3, 8, 9]:
        # restore store from start
        df['store_id'] = df.apply(lambda row: "_".join(row.id.split('_')[0:2]), axis=1)
    if level in [4, 5, 10, 12]:
        # restore category from start
        df['cat_id'] = df.apply(lambda row: row.id.split('_')[0], axis=1)
    if level in [5, 10, 12]:
        # restore department from start
        df['dept_id'] = df.apply(lambda row: "_".join(row.id.split('_')[0:2]), axis=1)
    if level in [6, 7, 11]:
        # restore category from second position
        df['cat_id'] = df.apply(lambda row: row.id.split('_')[1], axis=1)
    if level in [7, 11]:
        # restore department from second position
        df['dept_id'] = df.apply(lambda row: "_".join(row.id.split('_')[1:3]), axis=1)
    if level in [8, 9]:
        # restore department from third position
        df['dept_id'] = df.apply(lambda row: "_".join(row.id.split('_')[2:4]), axis=1)
    if level in [9]:
        # restore category from third position
        df['cat_id'] = df.apply(lambda row: row.id.split('_')[2], axis=1)
    if level in [10, 12]:
        # restore item id from start
        df['item_id'] = df.apply(lambda row: "_".join(row.id.split('_')[0:3]), axis=1)
    if level in [11]:
        # restore item id from second position
        df['item_id'] = df.apply(lambda row: "_".join(row.id.split('_')[1:4]), axis=1)
    if level in [12]:
        # restore state id from third position
        df['state_id'] = df.apply(lambda row: row.id.split('_')[3], axis=1)
    if level in [12]:
        # restore store from third position
        df['store_id'] = df.apply(lambda row: "_".join(row.id.split('_')[3:5]), axis=1)
    return df


def read_converted_sales(level, data_dir):
    # read converted sales
    converted_sales = pd.read_csv(data_dir + 'prep/converted_sales_level_{}.csv'.format(level), index_col=0)
    converted_sales = converted_sales.T

    # set index as column with name 'id'
    converted_sales.index.name = 'id'
    converted_sales = converted_sales.reset_index()

    # restore item_id/store_id/etc from the id
    restore_tags_converted_sales(converted_sales, level)

    return reduce_mem_usage(converted_sales)
