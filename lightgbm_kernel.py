"""

Code from public kernel LightGBM Poisson w/ Scaled Pinball Loss,
at https://www.kaggle.com/robertburbidge/lightgbm-poisson-w-scaled-pinball-loss

Converted to allow easy imports.
"""

# https://www.kaggle.com/girmdshinsei/for-japanese-beginner-with-wrmsse-in-lgbm
import warnings
# warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import gc
import os


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:  # columns毎に処理
        col_type = df[col].dtypes
        if col_type in numerics:  # numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


def read_data():
    print('Reading files...')
    calendar = pd.read_csv(os.environ['DATA_DIR'] + '/calendar.csv')
    calendar = reduce_mem_usage(calendar)
    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))
    #
    sell_prices = pd.read_csv(os.environ['DATA_DIR'] + '/sell_prices.csv')
    sell_prices = reduce_mem_usage(sell_prices)
    print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))
    #
    sales_train_val = pd.read_csv(os.environ['DATA_DIR'] + '/sales_train_validation.csv')
    print(
        'Sales train validation has {} rows and {} columns'.format(sales_train_val.shape[0], sales_train_val.shape[1]))
    #
    submission = pd.read_csv(os.environ['DATA_DIR'] + '/sample_submission.csv')
    #
    return calendar, sell_prices, sales_train_val, submission


def encode_categorical(df, cols):
    for col in cols:
        # Leave NaN as it is.
        le = LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)
    #
    return df


def simple_fe(data, DAYS_PRED=28):
    # demand features(過去の数量から変数生成)
    #
    for diff in [0, 1, 2]:
        shift = DAYS_PRED + diff
        data[f"shift_t{shift}"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(shift)
        )
    #
    for size in [7, 30, 60, 90, 180]:
        data[f"rolling_std_t{size}"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(size).std()
        )
    #
    for size in [7, 30, 60, 90, 180]:
        data[f"rolling_mean_t{size}"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(size).mean()
        )
    #
    data["rolling_skew_t30"] = data.groupby(["id"])["demand"].transform(
        lambda x: x.shift(DAYS_PRED).rolling(30).skew()
    )
    data["rolling_kurt_t30"] = data.groupby(["id"])["demand"].transform(
        lambda x: x.shift(DAYS_PRED).rolling(30).kurt()
    )
    #
    # price features
    # priceの動きと特徴量化（価格の変化率、過去1年間の最大価格との比など）
    #
    data["shift_price_t1"] = data.groupby(["id"])["sell_price"].transform(
        lambda x: x.shift(1)
    )
    data["price_change_t1"] = (data["shift_price_t1"] - data["sell_price"]) / (
        data["shift_price_t1"]
    )
    data["rolling_price_max_t365"] = data.groupby(["id"])["sell_price"].transform(
        lambda x: x.shift(1).rolling(365).max()
    )
    data["price_change_t365"] = (data["rolling_price_max_t365"] - data["sell_price"]) / (
        data["rolling_price_max_t365"]
    )
    #
    data["rolling_price_std_t7"] = data.groupby(["id"])["sell_price"].transform(
        lambda x: x.rolling(7).std()
    )
    data["rolling_price_std_t30"] = data.groupby(["id"])["sell_price"].transform(
        lambda x: x.rolling(30).std()
    )
    #
    # time features
    # 日付に関するデータ
    dt_col = "date"
    data[dt_col] = pd.to_datetime(data[dt_col])
    #
    attrs = [
        "year",
        "quarter",
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
    ]
    #
    for attr in attrs:
        dtype = np.int16 if attr == "year" else np.int8
        data[attr] = getattr(data[dt_col].dt, attr).astype(dtype)
    #
    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(np.int8)
    #
    return data


def weight_calc(data, weight_mat_csr):
    # calculate the denominator of RMSSE, and calculate the weight base on sales amount
    #
    sales_train_val = pd.read_csv(os.environ['DATA_DIR'] + '/sales_train_validation.csv')
    #
    d_name = ['d_' + str(i + 1) for i in range(1913)]
    #
    sales_train_val = weight_mat_csr * sales_train_val[d_name].values
    #
    # calculate the start position(first non-zero demand observed date) for each item / 商品の最初の売上日
    # 1-1914のdayの数列のうち, 売上が存在しない日を一旦0にし、0を9999に置換。そのうえでminimum numberを計算
    df_tmp = ((sales_train_val > 0) * np.tile(np.arange(1, 1914), (weight_mat_csr.shape[0], 1)))
    #
    start_no = np.min(np.where(df_tmp == 0, 9999, df_tmp), axis=1) - 1
    #
    # denominator of RMSSE / RMSSEの分母
    weight1 = np.sum((np.diff(sales_train_val, axis=1) ** 2), axis=1) / (1913 - start_no)
    #
    # calculate the sales amount for each item/level
    df_tmp = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
    df_tmp['amount'] = df_tmp['demand'] * df_tmp['sell_price']
    df_tmp = df_tmp.groupby(['id'])['amount'].apply(np.sum).values
    #
    weight2 = weight_mat_csr * df_tmp
    #
    weight2 = weight2 / np.sum(weight2)
    #
    del sales_train_val
    gc.collect()
    #
    return weight1, weight2


def make_wrmsse(NUM_ITEMS, weight_mat_csr, weight1, weight2):
    def wrmsse(preds, data):
        # actual obserbed values / 正解ラベル
        y_true = np.array(data.get_label())
        #
        # number of columns
        num_col = len(y_true) // NUM_ITEMS
        #
        # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) / 推論の結果が 1 次元の配列になっているので直す
        reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
        reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T
        #
        x_name = ['pred_' + str(i) for i in range(num_col)]
        x_name2 = ["act_" + str(i) for i in range(num_col)]
        #
        train = np.array(weight_mat_csr * np.c_[reshaped_preds, reshaped_true])
        #
        score = np.sum(
            np.sqrt(
                np.mean(
                    np.square(
                        train[:, :num_col] - train[:, num_col:])
                    , axis=1) / weight1) * weight2)
        #
        return 'wrmsse', score, False
    return wrmsse


def wrmsse_simple(preds, data, NUM_ITEMS, weight1, weight2):
    # actual obserbed values / 正解ラベル
    y_true = np.array(data.get_label())
    #
    # number of columns
    num_col = len(y_true) // NUM_ITEMS
    #
    # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) / 推論の結果が 1 次元の配列になっているので直す
    reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
    reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T
    #
    train = np.c_[reshaped_preds, reshaped_true]
    #
    weight2_2 = weight2[:NUM_ITEMS]
    weight2_2 = weight2_2 / np.sum(weight2_2)
    #
    score = np.sum(
        np.sqrt(
            np.mean(
                np.square(
                    train[:, :num_col] - train[:, num_col:])
                , axis=1) / weight1[:NUM_ITEMS]) * weight2_2)
    #
    return 'wrmsse', score, False


def agg_series(preds12, q):
    # preds12 contains 30490 series at level 12 aggregate these to get the other series
    preds12['id'] = preds12['item_id'] + "_" + preds12['store_id'] + "_" + q.replace('_', '.') + "_validation"
    # level 11: Unit sales of product x, aggregated for each State: 9,147
    preds11 = preds12.groupby(['date', 'item_id', 'state_id'], as_index=False)[q].sum()
    preds11['id'] = preds11['state_id'] + "_" + preds11['item_id'] + "_" + q.replace('_', '.') + "_validation"
    # level 10: Unit sales of product x, aggregated for all stores/states: 3,049
    preds10 = preds11.groupby(['date', 'item_id'], as_index=False)[q].sum()
    preds10['id'] = preds10['item_id'] + "_X_" + q.replace('_', '.') + "_validation"
    # level 9: Unit sales of all products, aggregated for each store and department: 70
    preds09 = preds12.groupby(['date', 'store_id', 'dept_id'], as_index=False)[q].sum()
    preds09['id'] = preds09['store_id'] + "_" + preds09['dept_id'] + "_" + q.replace('_', '.') + "_validation"
    # level 8: Unit sales of all products, aggregated for each store and category: 30
    preds08 = preds12.groupby(['date', 'store_id', 'cat_id'], as_index=False)[q].sum()
    preds08['id'] = preds08['store_id'] + "_" + preds08['cat_id'] + "_" + q.replace('_', '.') + "_validation"
    # level 7: Unit sales of all products, aggregated for each State and department: 21
    preds07 = preds12.groupby(['date', 'state_id', 'dept_id'], as_index=False)[q].sum()
    preds07['id'] = preds07['state_id'] + "_" + preds07['dept_id'] + "_" + q.replace('_', '.') + "_validation"
    # level 6: Unit sales of all products, aggregated for each State and category: 9
    preds06 = preds12.groupby(['date', 'state_id', 'cat_id'], as_index=False)[q].sum()
    preds06['id'] = preds06['state_id'] + "_" + preds06['cat_id'] + "_" + q.replace('_', '.') + "_validation"
    # level 5: Unit sales of all products, aggregated for each department: 7
    preds05 = preds12.groupby(['date', 'dept_id'], as_index=False)[q].sum()
    preds05['id'] = preds05['dept_id'] + "_X_" + q.replace('_', '.') + "_validation"
    # level 4: Unit sales of all products, aggregated for each category: 3
    preds04 = preds12.groupby(['date', 'cat_id'], as_index=False)[q].sum()
    preds04['id'] = preds04['cat_id'] + "_X_" + q.replace('_', '.') + "_validation"
    # level 3: Unit sales of all products, aggregated for each store: 10
    preds03 = preds12.groupby(['date', 'store_id'], as_index=False)[q].sum()
    preds03['id'] = preds03['store_id'] + "_X_" + q.replace('_', '.') + "_validation"
    # level 2: Unit sales of all products, aggregated for each State: 3
    preds02 = preds12.groupby(['date', 'state_id'], as_index=False)[q].sum()
    preds02['id'] = preds02['state_id'] + "_X_" + q.replace('_', '.') + "_validation"
    # level 1: Unit sales of all products, aggregated for all stores/states: 1
    preds01 = preds12.groupby(['date'], as_index=False)[q].sum()
    preds01['id'] = 'Total_X_' + q.replace('_', '.') + '_validation'
    preds = pd.concat([preds01, preds02, preds03,
                       preds04, preds05, preds06,
                       preds07, preds08, preds09,
                       preds10, preds11, preds12],
                      ignore_index=True, sort=True)
    return preds


# SPL for a single time series and single quantile
def SPL(ts, Qu, u, n, h):
    return (1.0 / h) * (n - 1) * sum((ts[n:n + h] - Qu) * u * (Qu <= ts[n:n + h]) + \
                                     (Qu - ts[n:n + h]) * (1 - u) * (Qu > ts[n:n + h])) / \
           sum(abs(ts[1:n] - ts[0:(n - 1)]))


# SPL for a single time series and multiple quantiles
def meanSPL(ts, Q, n, h):
    u = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
    #
    spls = np.zeros(len(u))
    for j in range(len(u)):
        spls[j] = SPL(ts, Q[j, :], u[j], n, h)
    #
    return np.mean(spls)


# run the calcn
def mymeanSPL(i, demandval, Ytrainvaldemand, n, h):
    Q = demandval[i * 28 * 9:(i + 1) * 28 * 9].reshape([9, 28])
    ts = Ytrainvaldemand[i * (n + h):(i + 1) * (n + h)]
    return meanSPL(ts, Q, n, h)
