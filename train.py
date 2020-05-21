"""
M5Forecast - Training
# Implementation of M5 Forecasting challenge on Kaggle, https://www.kaggle.com/c/m5-forecasting-accuracy/.

Created: 25 apr 2020
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
import tensorflow as tf

from flow import select_day_nums


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
                 inp_shape=(3244,), out_shape=(9,), categorical_features=None):
        """Initialization"""
        # Save settings
        self.df = df
        self.list_IDs = self.df.index
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ensure_all_samples = ensure_all_samples
        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.categorical_features = [c for c in categorical_features
                                     if c in features]

        # initialize indices
        self.indexes = None
        self.on_epoch_end()

        # calculate properties
        self.n = self.df.index.size

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
        list_IDs_temp = self.list_IDs[indexes]

        # Generate data
        x_batch, y_batch = self.__data_generation(list_IDs_temp)

        return x_batch, y_batch

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""

        # fill labels
        demand = self.df.loc[list_IDs_temp, 'demand'].values.astype(np.float32)
        y_batch = {'q%d' % d: demand for d in range(9)}

        # fill features
        x_batch = self.df.loc[list_IDs_temp, self.features]
        x_batch = pd.get_dummies(x_batch, columns=self.categorical_features)  # , dummy_na=True)

        # convert to floats
        x_batch = x_batch.astype(np.float32)
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
        print("Tracking {}".format(self.metric_names))

        self.val_metrics = {}
        self.val_metric_names = ['val_{}'.format(m) for m in self.metric_names]
        for m in self.val_metric_names:
            self.val_metrics[m] = []
        print("Tracking {}".format(self.val_metric_names))

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
            for i, m in self.val_metric_names:
                self.val_metrics[m] = val_losses[i]

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

    # first try: 7.687106850995075/0.0018890968224565899
    def plot(self, experimental_pinball_boost=3.671080060420607 / 0.03552745282649994, clear=True):
        if clear:
            clear_output()

        f, axes2d = plt.subplots(2, 2, figsize=(18, 12))

        # plot losses
        losses = self.train_metrics['loss']
        val_losses = self.val_metrics['val_loss']

        for i, axes in enumerate(axes2d):
            if i == 1:
                # experimental: convert normalised PL -> WSPL
                losses = np.array(losses) * experimental_pinball_boost
                val_losses = np.array(val_losses) * experimental_pinball_boost

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
