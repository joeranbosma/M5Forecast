# this is a basic neural network that:
# 0. loads a training dataset 
# 1. trains a neural network (model) on the dataset
# 1.1 The training consists of the following loop of: 1.1.1 to 1.1.3: 
# 1.1.1 setting weights in neurons in the network
# 1.1.2 predicting an output based on an input/datasample.
# 1.1.3 Performing backpropagation of the error to adjust the weights
# 2. loads a test set
# 3. Uses the trained model to perform a prediction/classification


# 0. loads a training dataset 
def get_train_data():
	pass

# 1. trains a neural network (model) on the dataset


# Code:
def get_model(inp_shape, quantiles):
    # clear previous sessions
    K.clear_session()

    inp = Input(inp_shape, name="input")
    x = inp
    x = Dense(16)(x)
    x = Dense(32)(x)
    x = Dense(64)(x)
    x = Dense(2)(x)  # represents mu, sigma
    x = DistributionLayer(quantiles=quantiles)(x)  # returns 7 points, one for each quantile
    out = x

    model = Model(inputs=inp, outputs=out)

    return model
	
	
# 2. loads a test set
def get_test_data():
	pass

# 3. Uses the trained model to perform a prediction/classification
def predict_output():
	pass