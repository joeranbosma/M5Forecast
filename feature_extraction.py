"""
M5Forecast - Feature extraction
# Implementation of M5 Forecasting challenge on Kaggle, https://www.kaggle.com/c/m5-forecasting-uncertainty/.

Created: 22 may 2020
"""

import numpy as np
import pandas as pd


# adapted from from https://www.kaggle.com/robertburbidge/lightgbm-poisson-w-scaled-pinball-loss
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
    if "sell_price" in data.columns:
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
