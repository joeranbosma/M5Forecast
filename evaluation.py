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
from flow import load_data, select_dates, sales_to_money, select_final_day, select_day_nums


def day_to_int(day, default=None):
    try:
        return int(day[2:])
    except (ValueError, TypeError) as e:
        return default


def convert_true_sales_to_quantiles(sales_true, aggregation_levels, postfix='_evaluation', verbose=1):
    sales_true = sales_true.copy()

    # Quantiles
    quantiles = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750,0.835, 0.975, 0.995]
    d_cols = select_day_nums(sales_true, as_int=False)

    sales_true_quantiles = pd.DataFrame()
    iterator = aggregation_levels.items()
    if verbose == 1:
        print("Converting true sales to quantile form")
    elif verbose == 2:
        iterator = tqdm(iterator, desc="True sales to quantiles")
    for level, groupby in iterator:
        if groupby:
            group = sales_true.groupby(groupby).sum()
        else:
            d_cols = select_day_nums(sales_true, as_int=False)
            group = sales_true[d_cols]

        group_ = pd.DataFrame()
        for quantile in quantiles:
            g_ = group.copy()
            vals = g_.index.values
            if level == 1:
                ids = ["Total_X_{:.3f}".format(quantile) + postfix for val in vals]
            if level in [2, 3, 4, 5, 10]:
                ids = [val + "_X_{:.3f}".format(quantile) + postfix for val in vals]
            if level in [6, 7, 8, 9]:
                ids = [val[0] + '_' + val[1] + "_{:.3f}".format(quantile) + postfix for val in vals]
            if level == 11:
                ids = [val[1] + '_' + val[0] + "_{:.3f}".format(quantile) + postfix for val in vals]
            if level == 12:
                ids = [val.replace('_validation', '') + "_{:.3f}".format(quantile) + postfix for val in vals]

            g_['quantile'] = quantile
            g_['level'] = level
            g_['id'] = ids

            group_ = group_.append(g_)

        sales_true_quantiles = sales_true_quantiles.append(group_)

    # convert quantiles to float
    conv_dict = {d_col: 'float64' for d_col in d_cols}
    sales_true_quantiles = sales_true_quantiles.astype(conv_dict)
    # sales_true_quantiles.columns = ['F%d' % int(i+1) if x =='d_1' + str(i+886) else x
    #                                 for i, x in enumerate(sales_true_quantiles.columns)]

    return sales_true_quantiles


class Referee(object):
    def __init__(self, sales_true, sales_train, prices, calendar, verbose=True):
        if verbose: print("Initializing Referee")
        self.sales_true = sales_true
        self.sales_train = sales_train
        self.h = sales_true.shape[1]
        self.n = sales_train.shape[1]

        self.quantiles = np.array([0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995])

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

        self.sales_true_quantiles = convert_true_sales_to_quantiles(sales_true, self.aggregation_levels)

        # Set number of aggregation levels
        self.K = len(self.aggregation_levels)  # 12 for full evaluation

        # Calculate weights (based on cumulative dollars per product, in last 28 days of train data)
        # Select final 28 days of training data
        last_train_day = select_final_day(sales_train)
        sales_train_final = select_dates(sales_train, day_end=last_train_day, num_days=28, include_metadata=True)

        # Try to find money spent pre-converted
        fn = os.environ['DATA_DIR'] + "/sales_train_money.csv"
        if os.path.exists(fn):
            sales_train_money = pd.read_csv(fn, index_col='id')
            sales_train_final = select_dates(sales_train_money, day_end=last_train_day, num_days=28, include_metadata=True)
        else:
            # Convert quantities sold to money spent
            sales_train_final = sales_to_money(sales_train_final, prices, calendar, verbose=verbose)

        # Calculate weights of each level
        if verbose: print("Calculating weights for each level...")
        self.weights = {}
        for level, groupby in self.aggregation_levels.items():
            self.weights[level] = self.calc_weights(sales_train_final, groupby=groupby)

        # Calculate scale of each level
        if verbose: print("Calculating scale for each level...")
        self.scales_WSPL = {}
        self.scales_WRMSSE = {}
        for level, groupby in self.aggregation_levels.items():
            scale_WSPL, scale_WRMSSE = self.calc_scale(groupby=groupby)
            self.scales_WSPL[level] = scale_WSPL
            self.scales_WRMSSE[level] = scale_WRMSSE

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
        # Calculate RWMSSE scale: 1/(n-1) * sum (Yt - Y_t-1) ^ 2
        # Calculate WSPL scale: 1/(n-1) * sum |Yt - Y_t-1|
        # agg level 3: 2.77 s ± 300 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # agg level 12: 3.91 s ± 921 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        """As done with RMSSE, the denominator of SPL is computed only for the time-periods for which the examined
        items/products are actively sold, i.e., the periods following the first non-zero demand observed for the
        series under evaluation. 
        """
        if groupby:
            Yt = self.sales_train.groupby(groupby).sum()
        else:
            day_cols = self.sales_train.filter(regex='d_').columns
            Yt = self.sales_train[day_cols]
        Yt1 = Yt.shift(1, axis=1)

        # Calculate number of sales since fist sale for each product (default is required when unit is not sold in
        # training period, which also makes its weight zero.)
        first_sold_day = Yt.replace(0, np.nan).apply(lambda x: day_to_int(x.first_valid_index(), default=-1), axis=1)
        n = select_final_day(Yt) - first_sold_day  # list of numbers since first sale

        # Calculate scales for WSPL / WRMSSE loss
        scale_WSPL = (np.abs(Yt - Yt1).sum(axis=1) / (n - 1))
        scale_WRMSSE = ((Yt - Yt1) ** 2).sum(axis=1) / (n - 1)
        return scale_WSPL, scale_WRMSSE

    def evaluate_WRMSSE(self, sales_pred):
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
        assert level is not 12, "Check if changing agg level groupby from None to ['store_id', 'item_id'] breaks this calculation"

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
            if scale is None: scale = self.scales_WRMSSE[level]
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

    def evaluate_WSPL(self, quantiles_pred):
        """Evaluate the Scaled Pinball Loss for a given set of predictions"""
        metrics = {}

        # Determine predicted levels
        predicted_levels = quantiles_pred.level.unique()

        # Calculate SPL for each level
        for level, groupby in self.aggregation_levels.items():
            if level in predicted_levels:
                # The groupby, weights and scale will be selected using the level
                metrics[level] = self.calc_SPL(quantiles_pred, level=level, groupby=groupby)

        WSPL = np.mean(list(metrics.values()))  # or sum and divide by self.K, take average over all aggregation levels
        metrics['WSPL'] = WSPL
        return metrics

    def calc_SPL(self, quantiles_pred, level=None, groupby=None, weights=None, scale=None, clip_zero=False):
        """Calculate the Scaled Pinball Loss for a given aggregation level"""
        if level:
            if groupby is None: groupby = self.aggregation_levels[level]
            if weights is None: weights = self.weights[level]
            if scale is None: scale = self.scales_WSPL[level]
        assert weights is not None, "Provide level or weights"
        assert scale is not None, "Provide level or scale"

        # Select the correct predictions and true sales based on the input level
        predictions = quantiles_pred[quantiles_pred['level'] == level]
        true_sales = self.sales_true_quantiles[self.sales_true_quantiles['level'] == level]

        if clip_zero:
            d_cols = select_day_nums(predictions, as_int=False)
            predictions[d_cols] = predictions[d_cols].clip(lower=0)

        # Make sure that both the predictions and the true sales have the same
        # id list, otherwise our calculation will go wrong
        predictions = predictions.sort_values('id')
        true_sales = true_sales.sort_values('id')

        # Convert to numpy array
        d_cols = select_day_nums(predictions, as_int=False)
        predictions = predictions[d_cols].values
        true_sales = true_sales[d_cols].values

        # Error
        err = true_sales - predictions

        # Number of rows
        Nlevel = predictions.shape[0]

        # Dummy array to save losses in
        losses = np.zeros(Nlevel // 9)
        for i in range(Nlevel // 9):
            indices = np.arange(i*9,(i+1)*9) # per set of 9, take indices
            subset = err[indices] # Take subset out of real set
            res = np.mean(np.sum(np.amax(np.array([self.quantiles * subset.T, (self.quantiles - 1) * subset.T]),axis=0),axis=0)) #compute PL of set
            losses[i] = res # Save resulting PL

        loss = np.sum(np.array(losses * weights)/np.array(self.h*scale)) # Calculate SPL of aggregate level

        return loss

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


class RapidReferee(object):
    """Use same scale and weight for each evaluation"""
    def __init__(self, sales_true, sales_train, prices, calendar, verbose=True):
        # initialize a Referee
        self.ref = Referee(sales_true, sales_train, prices, calendar, verbose=verbose)

    def evaluate(self, sales_pred, sales_true=None, mode='WSPL'):
        """Evaluate performance without re-calculating weights and scales
        mode can be either 'WSPL' or 'WRMSSE'.
        """
        # sales_true can be provided to update the true sales
        if sales_true is not None:
            self.ref.sales_true = sales_true

        # we will alter the df, so copy it
        sales_pred = sales_pred.copy()

        # rewrite columns to match the Referee's columns
        day_nums_pred = select_day_nums(sales_pred, as_int=False)
        day_nums_true = select_day_nums(self.ref.sales_true, as_int=False)

        pred_cols = list(sales_pred.columns)
        for source_day, target_day in zip(day_nums_pred, day_nums_true):
            idx = pred_cols.index(source_day)
            pred_cols[idx] = target_day

        sales_pred.columns = pred_cols
        if mode == 'WSPL':
            return self.ref.evaluate_WSPL(sales_pred)
        elif mode == 'WRMSSE':
            return self.ref.evaluate_WRMSSE(sales_pred)


class CrossValiDataGenerator:
    """
    Provide training and validation sets for cross-validation.
    The validation sets are 28 days each and are numbered in anti-chronological order,
    with the public leaderboard denoted as 0, the 28 days before that as 1, etc...

    When training a (large) network and you don't want to re-train the network before evaluating
    each fold, select the full training set associated with the number of folds you want to evaluate.
    That way, none of the samples in the validation folds are seen by the network during training.
    For evaluation, select the required number of days prior to the evaluation set to make predictions.
    """
    def __init__(self, sales_train_validation, train_size=-1, validation_size=28):
        """If train_size=-1, all days prior to the validation set will be the training set."""
        self.df = sales_train_validation
        self.train_size = train_size
        self.validation_size = validation_size
        self.final_day = 1941  # select_final_day(sales_train_validation)

    def get_train_val_split(self, fold=None, train_size=None):
        """Get training and validation sets for specified validation fold."""
        if train_size is None:
            train_size = self.train_size

        day_end = self.final_day - self.validation_size * fold
        val_df = select_dates(self.df, day_end=day_end, num_days=self.validation_size,
                              include_metadata=True)

        if train_size == -1:
            # select all days prior to validation set
            train_df = select_dates(self.df, day_start=1, day_end=day_end - self.validation_size,
                                    include_metadata=True)
        else:
            train_df = select_dates(self.df, num_days=train_size, day_end=day_end - self.validation_size,
                                    include_metadata=True)

        return train_df, val_df


def test_cv_generator(verbose=True):
    """Test the CrossValiDataGenerator"""
    # Load data
    calendar, sales_train_validation, sell_prices = load_data()

    # Set up generator
    cv_generator = CrossValiDataGenerator(sales_train_validation, train_size=28)
    train, _ = cv_generator.get_train_val_split(fold=10, train_size=-1)

    # Select train & test sets for ten folds
    val_days_seen = []
    for fold in range(1, 1 + 10):
        train_df, val_df = cv_generator.get_train_val_split(fold=fold)
        # select days of validation set
        d_list = select_day_nums(val_df)
        # assert none of these days were previously seen
        for d in d_list:
            assert (d not in val_days_seen), "Validation day already seen"

        # add to list of seen days
        val_days_seen.extend(d_list)

    if verbose:
        print("Validation days seen: ", val_days_seen)

    # the final 28*10 days should have been seen, not including the public leaderboard
    for d in range(1913-28*10+1, 1913+1):
        assert d in val_days_seen, "Validation day {} should have been seen".format(d)


if __name__ == "__main__":
    os.environ['DATA_DIR'] = 'data/'
    test_referee()
    test_cv_generator()