"""
M5Forecast - Training
# Implementation of M5 Forecasting challenge on Kaggle, https://www.kaggle.com/c/m5-forecasting-accuracy/.

Created: 25 apr 2020
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow.keras.callbacks import Callback

from flow import select_day_nums
from agent import AggregateAgent


class BatchCreator(object):
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
        self.n_features = len(features) * window_in
        self.n_labels = len(labels) * window_out

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
        x_batch = np.zeros(shape=(batch_size, self.n_features,))
        y_batch = np.zeros(shape=(batch_size, self.n_labels))

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

            x_batch[i] = self.df.loc[inp_idx, self.features].values.T.reshape(self.n_features)
            y_batch[i] = self.df.loc[val_idx, self.labels].values.T.reshape(self.n_labels)

        return x_batch, y_batch

    def flow(self):
        """returns a generator that will yield batches infinitely"""
        while True:
            for index in range(self.__len__()):
                batch_x, batch_y = self.__getitem__(index)
                yield batch_x, batch_y
            self.on_epoch_end()


class Logger(Callback):
    def __init__(self, ref, cv_generator, train_norm, folds, inp_shape=280,
                 preprocess=None, update_plot=True):
        super().__init__()
        self.ref = ref
        self.cv_generator = cv_generator
        self.train_norm = train_norm
        self.folds = folds
        self.inp_shape = inp_shape
        self.preprocess = preprocess
        self.update_plot = update_plot

        self.losses = []
        self.val_metrics = []
        self.best_wrmsse = np.inf
        self.best_model = None

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, batch, logs={}):
        mean_wrmsse, wrmsses = self.validate()
        self.val_metrics.append([len(self.losses), mean_wrmsse])
        if mean_wrmsse > self.best_wrmsse:
            self.best_wrmsse = mean_wrmsse
            self.best_model = self.model.get_weights()
        if self.update_plot:
            self.plot()

    def validate(self):
        ls = []

        for fold in self.folds:
            sales_train, sales_true = self.cv_generator.get_train_val_split(fold=fold, train_size=28)

            sales_true_aggregated = sales_true.groupby(['store_id']).sum()
            train_df, features, norm = self.preprocess(sales_train, norm=self.train_norm)

            # select days to predict
            val_day_nums = select_day_nums(sales_true_aggregated)
            # setup agent with trained model
            agent = AggregateAgent(model=self.model, train_norm=self.train_norm, inp_shape=self.inp_shape)

            # predict
            sales_pred = agent.predict(train_df, val_day_nums)

            store_WRMSSE = self.ref.calc_WRMSSE(sales_true=sales_true_aggregated, sales_pred=sales_pred, level=3)
            ls.append(store_WRMSSE)

        return np.mean(ls), ls

    def plot(self, clear=True):
        if clear:
            clear_output()

        N = len(self.losses)
        train_loss_plt, = plt.plot(range(0, N), self.losses)
        val_plt, = plt.plot(*np.array(self.val_metrics).T)
        plt.legend((train_loss_plt, val_plt),
                   ('training loss', 'validation WRMSSE'))
        plt.show()
