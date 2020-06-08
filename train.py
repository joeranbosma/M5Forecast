"""
M5Forecast - Training
# Implementation of M5 Forecasting challenge on Kaggle, https://www.kaggle.com/c/m5-forecasting-accuracy/.

Created: 25 apr 2020
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time, gc

from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
import tensorflow as tf

from flow import select_day_nums, evaluate_model
from model_builder import get_pinball_losses
from preprocess import read_and_preprocess_data


class WindowBatchCreator(Sequence):
    """Batch creator for M5Forecast - Accuracy challenge
    Expects a DataFrame with the days as index (d_num_start, .., d_num_end)

    - features: which columns to use for making predictions.
    - labels: which columns to predict.
    - window_in: number of input days for making predictions.
    - window_out: number of days to predict.
    - dilation: step size for training day selection.
    - lag: number of days between window_in days and window_out days.
    - shuffle: whether to shuffle the samples, where a sample consists of
               both the window_in days, window_out days.
    """

    def __init__(self, df, features, labels, window_in, window_out, dilation=1,
                 lag=0, batch_size=32, shuffle=True, ensure_all_samples=False):
        """Initialization"""
        # Save a reference to the df
        self.df = df
        self.features = features
        self.labels = labels

        # Save hyperparameters
        self.window_in = window_in
        self.window_out = window_out
        self.dilation = dilation
        self.lag = lag
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.ensure_all_samples = ensure_all_samples

        # Set up list of start indices for the validation set
        # From those the other indices will be calculated
        # Have 1 + (window_in - 1) * dilation training samples
        # Need `lag` days between training and validation samples
        train_day_span = 1 + (window_in - 1) * dilation
        start_val_day_min = min(select_day_nums(df, axis=0)) + train_day_span + lag
        start_val_day_max = max(select_day_nums(df, axis=0)) - window_out + 1
        self.list_start_val_days = np.arange(start_val_day_min, start_val_day_max + 1)

        # initialize indices
        self.indexes = None
        self.on_epoch_end()

        # calculate properties
        self.n = len(self.list_start_val_days)
        if isinstance(features, dict):
            self.n_features = {key: len(feats) for (key, feats) in features.items()}
        else:
            self.n_features = len(features)
        self.n_labels = len(labels)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        if self.ensure_all_samples:
            return int(np.ceil(self.n / self.batch_size))
        return self.n // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_start_val_days_temp = self.list_start_val_days[indexes]

        # Generate data
        x_batch, y_batch = self.__data_generation(list_start_val_days_temp)

        return x_batch, y_batch

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_start_val_days))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_start_val_days_temp):
        """Generates data containing batch_size samples"""

        # create batch placeholders
        batch_size = len(list_start_val_days_temp)
        if isinstance(self.features, dict):
            x_batch = {}
            for key, feats in self.features.items():
                x_batch[key] = np.zeros(shape=(batch_size, self.window_in, self.n_features[key]))
        else:
            x_batch = np.zeros(shape=(batch_size, self.window_in, self.n_features))
        y_batch = np.zeros(shape=(batch_size, self.window_out, self.n_labels))

        # fill batch
        for i, start_val_day in enumerate(list_start_val_days_temp):
            """
            start_val_day contains the first day of the evaluation set
            - final val day: start_val_day + window_out - 1
            - final input day: start_val_day - lag - 1
            - start input day: final input day - (window_in - 1) * dilation
            """
            # calculate day nums
            final_val_day = start_val_day + self.window_out - 1
            final_inp_day = start_val_day - self.lag - 1
            start_inp_day = final_inp_day - (self.window_in - 1) * self.dilation

            # print("Selecting train: {}-{} (step: {})".format(start_inp_day, final_inp_day, self.dilation))
            # print("Selecting eval: {}-{} (lag: {})".format(start_val_day, final_val_day, self.lag))

            # create lists of indices
            val_idx = ['d_%d' % d for d in range(start_val_day, final_val_day + 1)]
            inp_idx = ['d_%d' % d for d in range(start_inp_day, final_inp_day + 1, self.dilation)]

            # print("Train idx: {}".format(inp_idx))
            if isinstance(self.features, dict):
                for key, feats in self.features.items():
                    x_batch[key][i] = self.df.loc[inp_idx, feats].values
            else:
                x_batch[i] = self.df.loc[inp_idx, self.features].values
            y_batch[i] = self.df.loc[val_idx, self.labels].values

        return x_batch, y_batch

    def flow(self):
        """returns a generator that will yield batches infinitely"""
        while True:
            for index in range(self.__len__()):
                batch_x, batch_y = self.__getitem__(index)
                yield batch_x, batch_y
            self.on_epoch_end()


class BatchCreator(Sequence):
    """Batch creator for M5 Uncertainty challenge.
    - batch_size: number of samples per batch. Note: if ensure_all_samples is True,
                  the final batch size may be smaller.
    - shuffle: whether to shuffle the samples.
    - ensure_all_samples: whether to ensure all samples are yielded. If False (default),
                          the batch size is always constant.
    - inp_shape: input shape of how a single sample enters the neural network. This is without the batch size.
    - categorical_features: which columns to convert to one-hot encoding
    """

    def __init__(self, df, features, labels, batch_size=128, shuffle=True, ensure_all_samples=False,
                 inp_shape=(3244,), out_shape=(9,), categorical_features=None, check_nan=True):
        """Initialization"""
        # Save settings
        self.df = df
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ensure_all_samples = ensure_all_samples
        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.categorical_features = [c for c in categorical_features
                                     if c in features]
        self.check_nan = check_nan

        # calculate properties
        self.n = self.df.index.size

        # initialize indices
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        if self.ensure_all_samples:
            return int(np.ceil(self.n / self.batch_size))
        return self.n // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x_batch, y_batch = self.__data_generation(indexes)

        return x_batch, y_batch

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples"""

        # fill labels
        demand = self.df.iloc[indexes]['demand'].values.astype(np.float32)
        y_batch = {'q%d' % d: demand for d in range(9)}

        # fill features
        x_batch = self.df.iloc[indexes][self.features]
        x_batch = pd.get_dummies(x_batch, columns=self.categorical_features)  # , dummy_na=True)

        # convert to floats
        x_batch = x_batch.astype(np.float32)

        if self.check_nan:
            # replace nan with zero
            mask = x_batch.isna()
            x_batch[mask] = 0

        # convert to numpy array and return
        x_batch = x_batch.values

        return x_batch, y_batch

    def flow(self, epochs=None):
        """returns a generator that will yield batches infinitely"""
        epochs_done = 0
        while True:
            for index in range(self.__len__()):
                batch_x, batch_y = self.__getitem__(index)
                yield batch_x, batch_y

            # track nr. of epochs
            epochs_done += 1
            if epochs is not None and epochs_done == epochs:
                break  # stop yielding new elements

            # do on epoch end
            self.on_epoch_end()


class Logger(Callback):
    def __init__(self, val_batch_creator, model_dir=None, model_name="model", update_plot=True):
        super().__init__()
        self.val_batch_creator = val_batch_creator
        self.update_plot = update_plot

        # validation metrics
        self.val_x = []
        self.val_spl = []
        # save best model properties
        self.best_spl = np.inf
        self.best_model = None
        self.best_epoch = 0
        self.model_dir = model_dir
        self.model_name = model_name

        # initialize metrics
        self.train_metrics = {}
        self.metric_names = ['loss']
        self.metric_names.extend(['q{}_loss'.format(d) for d in range(9)])
        for m in self.metric_names:
            self.train_metrics[m] = []
        # print("Tracking {}".format(self.metric_names))

        self.val_metrics = {}
        self.val_metric_names = ['val_{}'.format(m) for m in self.metric_names]
        for m in self.val_metric_names:
            self.val_metrics[m] = []
        # print("Tracking {}".format(self.val_metric_names))

    def on_batch_end(self, batch, logs={}):
        # log training metrics
        for m in self.metric_names:
            if m in logs.keys():
                self.train_metrics[m].append(logs.get(m))

    def on_epoch_end(self, batch, logs={}):
        num_train_steps = len(self.train_metrics['loss'])
        timestamp = time.strftime('%Y-%m-%d_%H%M', time.localtime())

        if self.model_dir:
            self.model.save(self.model_dir + '{}_{}_{}_steps.h5'.format(
                self.model_name, timestamp, num_train_steps))

        # calculate normalised validation PL
        if 'val_loss' in logs.keys():
            for m in self.val_metric_names:
                self.val_metrics[m].append(logs.get(m))
        else:
            # evaluate validation set
            val_losses = self.model.evaluate(self.val_batch_creator.flow(),
                                             steps=self.val_batch_creator.__len__())
            for i, m in enumerate(self.val_metric_names):
                self.val_metrics[m].append(val_losses[i])

        self.val_x.append(num_train_steps)
        spl = self.val_metrics['val_loss'][-1]

        if spl < self.best_spl:
            self.best_spl = spl
            self.best_model = self.model.get_weights()
            self.best_epoch = len(self.val_spl)
        if self.update_plot:
            self.plot()

    def validate(self):
        pass

    def plot(self, clear=True):
        if clear:
            clear_output()

        f, axes = plt.subplots(1, 2, figsize=(18, 6))

        # plot losses
        losses = self.train_metrics['loss']
        val_losses = self.val_metrics['val_loss']

        ax = axes[0]
        ax.plot(range(1, 1 + len(losses)), losses, label='Train')
        if len(val_losses):
            ax.plot(self.val_x, val_losses, '.-', label='Validation')
        ax.set_xlabel("Step")
        ax.set_ylabel(r"normalised PL")
        ax.set_title("Loss")
        ax.set_ylim(0)

        # plot final losses
        ax = axes[1]
        N = len(losses)
        n = min(501, max(100, N - 100))
        ax.plot(range(1 + N - n, 1 + N), losses[-n:], label='Train')
        if len(val_losses):
            indexer = [x > (N - n) for x in self.val_x]
            ax.plot(np.array(self.val_x)[indexer], np.array(val_losses)[indexer], '.-', label='Validation')
        ax.set_xlabel("Step")
        ax.set_ylabel(r"normalised PL")
        ax.set_title("Loss final {} steps".format(n))
        ax.set_ylim(0)

        for ax in axes:
            ax.legend()

        plt.show()


class WindowLogger(Callback):
    def __init__(self, ref, cv_generator, prices=None, calendar=None, train_norm=None, features=None, labels=None,
                 agent=None, folds=None, window_in=28, preprocess_func=None, update_plot=True,
                 plot_loss_max=None):
        super().__init__()
        self.ref = ref
        self.cv_generator = cv_generator
        self.prices = prices
        self.calendar = calendar
        self.train_norm = train_norm
        self.features = features
        self.labels = labels
        self.folds = folds
        self.agent = agent
        self.window_in = window_in
        self.preprocess_func = preprocess_func
        self.update_plot = update_plot
        self.plot_loss_max = plot_loss_max

        self.losses = []
        self.val_metrics = []
        self.best_wrmsse = np.inf
        self.best_model = None

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, batch, logs={}):
        mean_wrmsse, wrmsses = self.validate()
        self.val_metrics.append([len(self.losses), mean_wrmsse])
        if mean_wrmsse < self.best_wrmsse:
            self.best_wrmsse = mean_wrmsse
            self.best_model = self.model.get_weights()
        if self.update_plot:
            self.plot()

    def validate(self):
        ls = []

        for fold in self.folds:
            sales_train, sales_true = self.cv_generator.get_train_val_split(fold=fold, train_size=self.window_in)

            sales_true_aggregated = sales_true.groupby(['store_id']).sum()
            train_df, norm = self.preprocess_func(sales_train, prices=self.prices,
                                                  calendar=self.calendar, norm=self.train_norm)

            # select days to predict
            val_day_nums = select_day_nums(sales_true_aggregated)
            sales_pred = self.agent.predict(train_df, val_day_nums)

            store_WRMSSE = self.ref.calc_WRMSSE(sales_true=sales_true_aggregated, sales_pred=sales_pred.T,
                                                groupby=None, weights=self.ref.weights[3], scale=self.ref.scales[3])
            ls.append(store_WRMSSE)

        return np.mean(ls), ls

    def plot(self, clear=True):
        if clear:
            clear_output()

        N = len(self.losses)
        train_loss_plt, = plt.plot(range(0, N), self.losses)
        val_plt, = plt.plot(*np.array(self.val_metrics).T)
        if min(self.losses) < self.plot_loss_max:
            plt.ylim(top=self.plot_loss_max, bottom=0)
        plt.legend((train_loss_plt, val_plt),
                   ('training loss', 'validation WRMSSE'))
        plt.show()


def make_loss(ref, train_norm):
    # calculate scaling constant for each series
    scaling = ref.weights[3] * train_norm / (ref.scales[3] ** (1/2))
    # convert to  array
    scaling = tf.Variable(scaling.values, dtype=tf.float32)

    def WRMSSE_store(y_true, y_pred):
        # calculate squared prediction error
        SE = K.square(y_pred - y_true)
        # Calculate mean of daily prediction errors, so keep list of series
        MSE = K.mean(SE, axis=0)
        # apply scaling
        scaled_MSE = MSE * scaling
        # sum
        return K.sum(scaled_MSE)
    return WRMSSE_store


def plot_confidence_series(quantile_preds, quantiles, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    x = np.arange(len(quantile_preds[0.5]))

    # plot median as thick line
    ax.plot(x, quantile_preds[0.5], 'k-', linewidth=2, label="Median")

    # plot true sales
    ax.plot(x, quantile_preds['true'], 'g-', linewidth=3, label='True')

    # plot confidence intervals
    conf_labels = ['50%', '67%', '95%', '99%']
    for i in range(4):
        q1 = quantiles[i]
        q2 = quantiles[-i - 1]
        ax.fill_between(x, quantile_preds[q1], quantile_preds[q2], color='C0', alpha=(i + 1) * 0.2,
                        label=conf_labels[i])

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0)
    ax.set_xlabel("Predicted day")
    ax.set_ylabel("Predicted number of sales")
    ax.set_title(quantile_preds['label'])
    ax.legend()


def plot_some_confidence_intervals(df, val_batch_creator, level, quantiles, data_dir='data/', num=9, plot_shape=(3, 3)):
    indices = range(num)
    norm = pd.read_csv(data_dir + 'prep/norm_level_{}.csv'.format(level))

    f, axes = plt.subplots(nrows=plot_shape[0], ncols=plot_shape[1], figsize=(18, 6 * plot_shape[0]))

    for idx, ax in zip(indices, np.ravel(axes)):
        quantile_preds = {}
        d_cols = select_day_nums(df, as_int=False)

        for i, q in enumerate(quantiles):
            selected_series = df.loc[df['quantile'] == q].iloc[idx]
            quantile_preds[q] = selected_series[d_cols].values.astype(float)

        series_id = "_".join(selected_series['id'].split('_')[0:-2])  # e.g. FOODS_1_010_X_0.995_evaluation
        true_sales = val_batch_creator.df.loc[(val_batch_creator.df['id'] == series_id), 'demand']
        series_norm = norm.loc[norm['id'] == series_id].norm.values[0]
        quantile_preds['true'] = (true_sales * series_norm).values
        quantile_preds['label'] = series_id

        # plot
        plot_confidence_series(quantile_preds, quantiles=quantiles, ax=ax)

    plt.tight_layout()
    plt.show()


def prepare_training(data, features, labels, available_cat_features, batch_size=1024):
    # going to evaluate with the last 28 days
    x_train = data[data['date'] <= '2016-03-27']
    x_val = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]

    # make batch creators
    labels = ['demand']

    def get_generators(bs=1024):
        train_bc = BatchCreator(x_train, features, labels, categorical_features=available_cat_features,
                                batch_size=bs, check_nan=False)
        val_bc = BatchCreator(x_val, features, labels, shuffle=False, ensure_all_samples=True,
                              categorical_features=available_cat_features, batch_size=bs, check_nan=False)

        return train_bc, val_bc

    train_batch_creator, val_batch_creator = get_generators(bs=batch_size)

    # determine model input shape
    x, y = next(train_batch_creator.flow())
    INP_SHAPE = x[0].shape

    # make losses
    losses = get_pinball_losses()

    return train_batch_creator, val_batch_creator, get_generators, INP_SHAPE, losses


def add_lgb_predictions(data, level, features, lgb_prediction_dir):
    # read predictions
    lgb_predictions = pd.read_csv(lgb_prediction_dir + 'predictions_level{}.csv'.format(level),
                                  index_col=0)

    # drop 'demand' column and convert date to pandas datetime
    if 'demand' in lgb_predictions.columns:
        lgb_predictions.drop(columns=['demand'], inplace=True)
    lgb_predictions.date = pd.to_datetime(lgb_predictions.date)

    # merge lgb's predictions with the data
    data = pd.merge(data, lgb_predictions, how='left', on=['id', 'date'])

    # add feature 'lgb_pred' to the feature list
    features.append('lgb_pred')

    return data, features


def perform_training_scheme(level, model, warmup_batch_size, finetune_batch_size, ref, calendar, quantiles=None,
                            data_dir='data/', model_dir='models/uncertainty/', warmup_lr_list=None,
                            finetune_lr_list=None, warmup_epochs=10, finetune_epochs=10, lgb_prediction=False,
                            lgb_prediction_dir=None, model_name="stepped_lr", validation_steps=None,
                            augment_events=False, verbose=True):
    if warmup_lr_list is None:
        warmup_lr_list = [1e-5, 1e-4, 1e-3, 2e-3, 3e-3, 1e-3]
    if finetune_lr_list is None:
        finetune_lr_list = [2e-3, 3e-3, 1e-3, 3e-4, 1e-4]
    if quantiles is None:
        quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
    print("Starting level {}..".format(level)) if verbose else None

    # read data
    data, features, available_cat_features = read_and_preprocess_data(level=level, augment_events=augment_events)
    if lgb_prediction:
        print("Adding lgb prediction as feature") if verbose else None
        data, features = add_lgb_predictions(data, level, features, lgb_prediction_dir)

    # setup for training
    batch_size = warmup_batch_size[level]
    labels = ['demand']
    train_batch_creator, val_batch_creator, get_generators, INP_SHAPE, losses = prepare_training(
        data, features, labels, available_cat_features, batch_size=batch_size)

    # compile model and initialize logger
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=losses)
    logger = Logger(val_batch_creator)

    # train model: warm-up
    lr_list = warmup_lr_list[0:3]
    for lr_block in lr_list:
        # set lr (without recompiling and losing momentum)
        def lr_scheduler(epoch, lr):
            return lr_block

        lr_callback = LearningRateScheduler(lr_scheduler, verbose=1)

        # train model
        val_steps = validation_steps if validation_steps is not None else val_batch_creator.__len__()
        history = model.fit(train_batch_creator.flow(), epochs=warmup_epochs, steps_per_epoch=100,
                            validation_data=val_batch_creator.flow(), validation_steps=val_steps,
                            callbacks=[lr_callback, logger])

    # evaluate
    metrics, df = evaluate_model(model, ref, val_batch_creator, calendar, quantiles, data_dir, level)
    metrics1 = metrics

    # save warm-up result
    model.save_weights(model_dir + 'level{}_{}_part1_WSPL{:.2e}.h5'.format(level, model_name, metrics['WSPL']))

    # train model: continued
    lr_list = warmup_lr_list[3:6]
    for lr_block in lr_list:
        # set lr (without recompiling and losing momentum)
        def lr_scheduler(epoch, lr):
            return lr_block

        lr_callback = LearningRateScheduler(lr_scheduler, verbose=1)

        # train model
        val_steps = validation_steps if validation_steps is not None else val_batch_creator.__len__()
        history = model.fit(train_batch_creator.flow(), epochs=warmup_epochs, steps_per_epoch=100,
                            validation_data=val_batch_creator.flow(), validation_steps=val_steps,
                            callbacks=[lr_callback, logger])

    # evaluate
    metrics, df = evaluate_model(model, ref, val_batch_creator, calendar, quantiles, data_dir, level)
    metrics2 = metrics

    # save continued result
    model.save_weights(model_dir + 'level{}_{}_part2_WSPL{:.2e}.h5'.format(level, model_name, metrics['WSPL']))

    # fine-tune
    batch_size = finetune_batch_size[level]
    train_batch_creator, val_batch_creator = get_generators(batch_size)

    lr_list = finetune_lr_list

    for lr_block in lr_list:
        # set lr (without recompiling and losing momentum)
        def lr_scheduler(epoch, lr):
            return lr_block

        lr_callback = LearningRateScheduler(lr_scheduler, verbose=1)

        # train model
        val_steps = validation_steps if validation_steps is not None else val_batch_creator.__len__()
        history = model.fit(train_batch_creator.flow(), epochs=finetune_epochs, steps_per_epoch=100,
                            validation_data=val_batch_creator.flow(), validation_steps=val_steps,
                            callbacks=[lr_callback, logger])

    # calculate WSPL and save metrics
    metrics, df = evaluate_model(model, ref, val_batch_creator, calendar, quantiles, data_dir, level)
    metrics3 = metrics

    # save fine-tuned model
    model.save_weights(model_dir + 'level{}_{}_part3_WSPL{:.2e}.h5'.format(level, model_name, metrics['WSPL']))

    return model, logger, metrics1, metrics2, metrics3


def get_chronological_train_val_split(data, fold=1, num_folds=5, verbose=True):
    # get chronological train/val split
    # fold 1 has the final 376 validation days
    # fold 2 has days -752:-376, etc.

    # setup
    # substract one second to include first day in validation set of fold 5
    day_start = data.date.min() - pd.Timedelta(seconds=1)
    day_end = data.date.max()
    num_days = day_end - day_start
    num_days_val = (num_days / num_folds)

    # select validation set
    val_start = day_end - fold * num_days_val
    val_end = day_end - (fold - 1) * num_days_val

    print("Selecing validation days between {} and {}".format(val_start, val_end)) if verbose else None

    # split
    val_mask = (data.date > val_start) & (data.date <= val_end)
    train = data[~val_mask]
    val = data[val_mask]

    return train, val


def get_train_val_slit(level, fold, augment_events=False, verbose=True):
    # read data
    data, features, available_cat_features = read_and_preprocess_data(level=level, verbose=verbose,
                                                                      augment_events=augment_events)

    # leave final days alone
    test = data[data['date'] > '2016-03-27']
    data = data[data['date'] <= '2016-03-27']

    # chronological train / val split
    train, val = get_chronological_train_val_split(data, fold=fold, verbose=verbose)

    return train, val, test, features


def train_lightgbm_model(level, fold=1, params={}, model_dir='models/uncertainty/',
                         model_name="lightgbm", augment_events=False, verbose=True,
                         num_boost_round=2500, early_stopping_rounds=50, verbose_eval=50):
    # only require lightgbm to be installed when calling this function
    import lightgbm as lgb

    # read data
    train, val, test, features = get_train_val_slit(level, fold, augment_events=augment_events)

    # make lgb datasets
    labels = ['demand']
    train_set = lgb.Dataset(train[features], train[labels])
    val_set = lgb.Dataset(val[features], val[labels])

    # cleanup memory
    del train
    gc.collect()

    # perform training
    evals_result = {}  # to record eval results for plotting
    model = lgb.train(params, train_set, num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds,
                      valid_sets=[val_set], verbose_eval=verbose_eval,  # fobj="mae",#feval = "mae",
                      evals_result=evals_result)

    model.save_model(model_dir + model_name + "-level{}-fold{}.txt".format(level, fold))
    ax = lgb.plot_metric(evals_result, metric='l1')
    plt.show()

    return model, evals_result, val


def lightgbm_pred_to_df(y_pred, df):
    ids = df['id'].unique()
    day_start = np.datetime64(df.date.min().date())
    day_end = np.datetime64(df.date.max().date())
    num_days = (day_end - day_start + 1).astype(int)

    y_pred_df = pd.DataFrame.from_dict({
        'id': np.repeat(ids, num_days),
        'date': np.tile(np.arange(day_start, day_end + 1), len(ids)),
        'lgb_pred': y_pred,
        'demand': df['demand'].values,
    })
    # y_pred_df['item_id'] = y_pred_df['id'].map(lambda x: '_'.join(x.split('_')[0:3]))
    # y_pred_df['dept_id'] = y_pred_df['id'].map(lambda x: '_'.join(x.split('_')[0:2]))
    # y_pred_df['cat_id'] = y_pred_df['id'].map(lambda x: '_'.join(x.split('_')[0:1]))
    # y_pred_df['store_id'] = y_pred_df['id'].map(lambda x: '_'.join(x.split('_')[3:5]))
    # y_pred_df['state_id'] = y_pred_df['id'].map(lambda x: '_'.join(x.split('_')[3:4]))

    return y_pred_df


def plot_lgb_metrics(evals_result, title):
    score = evals_result['valid_0']['l1']
    f, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(score)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L1 loss")
    ax.set_title(title)
    ax.set_xlim(0)
    ax.text(0.95, 0.95, "Best val. score: {:.5f}".format(min(score)),
            ha='right', va='top', transform=ax.transAxes)
    plt.show()


def plot_result_list(result_list, title_list):
    for evals_result, title in zip(result_list, title_list):
        plot_lgb_metrics(evals_result, title)
