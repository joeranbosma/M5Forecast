"""
M5Forecast - Model builder
# Implementation of M5 Forecasting challenge on Kaggle, https://www.kaggle.com/c/m5-forecasting-uncertainty/.

Created: 22 may 2020
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, LeakyReLU
from tensorflow.keras.layers import Flatten, Input, BatchNormalization, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Reshape, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam


# from Daniel Sch., at:
# https://stackoverflow.com/questions/43151694/define-pinball-loss-function-in-keras-with-tensorflow-backend
def create_pinball_loss(tau=0.5):
    def pinball_loss(y_true, y_pred):
        err = y_true - y_pred
        return K.mean(K.maximum(tau * err, (tau - 1) * err), axis=-1)
    return pinball_loss


def get_pinball_losses(quantiles=None):
    if quantiles is None:
        quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]

    losses = {'q' + str(i): create_pinball_loss(tau=q) for (i, q) in enumerate(quantiles)}
    return losses


# Lambda layer: https://blog.paperspace.com/working-with-the-lambda-layer-in-keras/
def get_custom_layer(sigma_coef):
    def custom_layer(tensor):
        tensor1 = tensor[0]
        tensor2 = tensor[1]
        return tensor1 + sigma_coef * tensor2

    return custom_layer


def get_simple_dist_model(inp_shape, num_nodes=64, final_activation="linear", sigma_coefs=None, clear_session=True):
    if sigma_coefs is None:
        sigma_coefs = [-2.57583, -1.95996, -0.974114, -0.674, 0, 0.674, 0.9741114, 1.95996, 2.57583]

    if clear_session:
        # clear previous sessions
        K.clear_session()

    inp = Input(inp_shape, name="input")
    x = inp
    x = Dense(num_nodes, activation="relu")(x)
    x = Dense(num_nodes, activation="relu")(x)
    x = Dense(num_nodes, activation="relu")(x)

    mu = Dense(1, activation=final_activation)(x)  # represents mu
    sigma = Dense(1, activation=final_activation)(x)  # represents sigma

    outs = []

    for i, sigma_coef in enumerate(sigma_coefs):
        custom_layer = get_custom_layer(sigma_coef=sigma_coef)
        out_q = Lambda(custom_layer, name="q{}".format(i))([mu, sigma])
        outs.append(out_q)

    model = Model(inputs=inp, outputs=outs)

    return model


def get_simple_dense_model(inp_shape, num_nodes=64, num_layers=3, bottleneck_nodes=None,
                           final_activation="linear", clear_session=True):
    if clear_session:
        # clear previous sessions
        K.clear_session()

    inp = Input(inp_shape, name="input")
    x = inp

    # add dense layers
    for i in range(num_layers):
        x = Dense(num_nodes, activation="relu")(x)

    if bottleneck_nodes is not None:
        x = Dense(bottleneck_nodes, activation="relu")(x)  # represents mu, sigma

    out_q0 = Dense(1, name="q0", activation=final_activation)(x)
    out_q1 = Dense(1, name="q1", activation=final_activation)(x)
    out_q2 = Dense(1, name="q2", activation=final_activation)(x)
    out_q3 = Dense(1, name="q3", activation=final_activation)(x)
    out_q4 = Dense(1, name="q4", activation=final_activation)(x)
    out_q5 = Dense(1, name="q5", activation=final_activation)(x)
    out_q6 = Dense(1, name="q6", activation=final_activation)(x)
    out_q7 = Dense(1, name="q7", activation=final_activation)(x)
    out_q8 = Dense(1, name="q8", activation=final_activation)(x)

    model = Model(inputs=inp, outputs=[out_q0, out_q1, out_q2, out_q3, out_q4, out_q5, out_q6, out_q7, out_q8])

    return model


def get_extended_custom_layer(sigma_coefs, i):
    def custom_layer(tensor):
        tensor1 = tensor[0]
        tensor2 = tensor[1]
        kurtosis = tensor[2]
        skewness = tensor[3]

        sigma_coef_max = np.max(sigma_coefs)
        if (i in [1, 2, 3, 5, 6, 7]):

            # apply skewness (-1,1) to coefficient
            mod_sigma_coefs = sigma_coefs[i] + (sigma_coef_max - sigma_coefs[i]) * skewness

            # shift x-coordinates towards mean for normalised kurtosis of 1
            return tensor1 + mod_sigma_coefs * (1 - kurtosis) * tensor2;
        else:
            # apply skewness
            if (i == 4):
                # apply skewness
                spoofed_sigma_coefs_5 = -sigma_coefs[3] + (sigma_coef_max + sigma_coefs[3]) * skewness
                spoofed_sigma_coefs_3 = sigma_coefs[3] + (sigma_coef_max - sigma_coefs[3]) * skewness
                mod_sigma_coefs = spoofed_sigma_coefs_3 + (spoofed_sigma_coefs_5 - spoofed_sigma_coefs_3) / 2

                # shift x-coordinates towards mean for normalised kurtosis of 1
                return tensor1 + mod_sigma_coefs * (1 - kurtosis) * tensor2;

            # keep outer(0,8) and middle quantile(4) x-coordinate the same for kurtosis
            else:
                return tensor1 + sigma_coefs[i] * tensor2

    return custom_layer


# dummy function to test the implemented skewness implementation
def apply_skewness(sigma_coefs, skewness):
    mod_sigma_coefs = np.zeros((len(sigma_coefs)), dtype=float)
    sigma_coef_max = np.max(sigma_coefs)
    for i in range(0, len(sigma_coefs)):
        if (i in [1, 2, 3, 5, 6, 7]):
            mod_sigma_coefs[i] = sigma_coefs[i] + (sigma_coef_max - sigma_coefs[i]) * skewness

    # center new mean by computing sigma_coefs[5] using only sigma_coefs[3] (and 4)
    spoofed_sigma_coefs_5 = -sigma_coefs[3] + (sigma_coef_max + sigma_coefs[3]) * skewness
    mod_sigma_coefs[4] = mod_sigma_coefs[3] + (spoofed_sigma_coefs_5 - mod_sigma_coefs[3]) / 2
    return mod_sigma_coefs


def get_extended_dist_model(inp_shape, sigma_coefs, clear_session=True):
    if sigma_coefs is None:
        sigma_coefs = [-2.57583, -1.95996, -0.974114, -0.674, 0, 0.674, 0.9741114, 1.95996, 2.57583]

    if clear_session:
        # clear previous sessions
        K.clear_session()

    inp = Input(inp_shape, name="input")
    x = inp
    x = Dense(16)(x)
    x = Dense(32)(x)
    x = Dense(64)(x)

    mu = Dense(1)(x)  # represents mu
    sigma = Dense(1)(x)  # represents sigma
    kurtosis = Dense(1)(x)  # represents kurtosis
    skewness = Dense(1)(x)  # represents skewness
    outs = []

    print(f'skewed sigma_coefs={apply_skewness(sigma_coefs, 0.4)}')
    print(f'skewed sigma_coefs={apply_skewness(sigma_coefs, -0.4)}')

    for i, sigma_coef in enumerate(sigma_coefs):
        custom_layer = get_extended_custom_layer(sigma_coefs=sigma_coefs, i=i)
        out_q = Lambda(custom_layer, name="q{}".format(i))([mu, sigma, kurtosis, skewness])
        outs.append(out_q)

    model = Model(inputs=inp, outputs=outs)

    return model


def get_variable_dist_model(inp_shape, sigma_coefs, num_nodes=64, final_activation="exponential", clear_session=True):
    if clear_session:
        # clear previous sessions
        K.clear_session()

    inp = Input(inp_shape, name="input")
    x = inp
    x = Dense(num_nodes, activation='relu')(x)
    x = Dense(num_nodes, activation='relu')(x)
    x = Dense(num_nodes, activation='relu')(x)

    mu = Dense(1, activation=final_activation)(x)  # represents mu
    sigma = Dense(1, activation=final_activation)(x)  # represents sigma
    kurtosis = Dense(1)(x)  # represents kurtosis
    skewness = Dense(1)(x)  # represents skewness
    outs = []

    print(f'skewed sigma_coefs={apply_skewness(sigma_coefs, 0.4)}')
    print(f'skewed sigma_coefs={apply_skewness(sigma_coefs, -0.4)}')

    for i, sigma_coef in enumerate(sigma_coefs):
        custom_layer = get_extended_custom_layer(sigma_coefs=sigma_coefs, i=i)
        out_q = Lambda(custom_layer, name="q{}".format(i))([mu, sigma, kurtosis, skewness])
        outs.append(out_q)

    model = Model(inputs=inp, outputs=outs)

    return model


def get_direct_custom_layer(i):
    def custom_layer(tensor):
        # unpack input
        qm = tensor[0]

        # parameters
        alpha = tensor[1]
        beta = tensor[2]
        gamma = tensor[3]
        delta = tensor[4]
        epsilon = tensor[5]
        zeta = tensor[6]
        eta = tensor[7]
        theta = tensor[8]

        if i == 0:
            return alpha * qm
        elif i == 1:
            return (alpha + gamma * beta - alpha * beta * gamma) * qm
        elif i == 2:
            return (alpha + beta - alpha * beta) * qm
        elif i == 3:
            return (alpha + beta - alpha * beta + epsilon - (alpha + beta - alpha * beta) * epsilon) * qm
        elif i == 5:
            return (1 + eta * zeta * delta) * qm
        elif i == 6:
            return (1 + zeta * delta) * qm
        elif i == 7:
            return (1 + zeta * delta + theta * delta + theta * eta * zeta * delta) * qm
        elif i == 8:
            return (1 + delta) * qm

    return custom_layer


def get_direct_dist_model(inp_shape, num_nodes=256, final_activation=None, clear_session=True):
    # final activation parameter only for compatibility

    if clear_session:
        # clear previous sessions
        K.clear_session()

    inp = Input(inp_shape, name="input")
    x = inp
    x = Dense(num_nodes, activation='relu')(x)
    x = Dense(num_nodes, activation='relu')(x)
    x = Dense(num_nodes, activation='relu')(x)

    # direct prediction of median
    qm = Dense(1, name="q4", activation="exponential")(x)

    # setup parameters
    alpha = Dense(1, name="alpha", activation="sigmoid")(x)
    beta = Dense(1, name="beta", activation="sigmoid")(x)
    gamma = Dense(1, name="gamma", activation="sigmoid")(x)
    delta = Dense(1, name="delta", activation="exponential")(x)
    epsilon = Dense(1, name="epsilon", activation="sigmoid")(x)
    zeta = Dense(1, name="zeta", activation="sigmoid")(x)
    eta = Dense(1, name="eta", activation="sigmoid")(x)
    theta = Dense(1, name="theta", activation="sigmoid")(x)

    outs = []

    for i in range(9):
        if i == 4:
            out_q = qm
        else:
            custom_layer = get_direct_custom_layer(i)
            params = [qm, alpha, beta, gamma, delta, epsilon, zeta, eta, theta]
            out_q = Lambda(custom_layer, name="q{}".format(i))(params)
        outs.append(out_q)

    model = Model(inputs=inp, outputs=outs)

    return model
