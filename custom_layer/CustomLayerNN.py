# this is a basic neural network that:
# 0. loads a training dataset 
# 0.1 Perform a preprocessing
# 1. Create model
# 1.1 Call custom layer
# 2. trains a neural network (model) on the dataset
# 2.1 The training consists of the following loop of: 1.1.1 to 1.1.3: 
# 2.1.1 setting weights in neurons in the network
# 2.1.2 predicting an output based on an input/datasample.
# 2.1.3 Performing backpropagation of the error to adjust the weights
# 3. loads a test set
# 4. Uses the trained model to perform a prediction/classification

# basic imports
import os
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt

# tensorflow imports
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Reshape, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class custom_layer_nn:

	def __init__(self):
		print("Hello world")
		pass

	# 0. loads a training dataset 
	def get_train_data(self):
		train_df = pd.read_csv("features.csv", index_col=0)
		target_df = pd.read_csv("targets.csv", index_col=0)
		
		print('Training, and target, data shape : ', train_df.shape, target_df.shape)
		return train_df, target_df
	
	# preprocess data
	def preproces(self,train_df):
		train_X = train_df.reshape(-1, 28,28, 1) # if necessesary
		return train_df
	
	# 1. Create model
	def get_model(self, inp_shape, quantiles):
		# clear previous sessions
		K.clear_session()
		
		inp = Input(inp_shape, name="input")
		x = inp
		x = Dense(16)(x)
		x = Dense(32)(x)
		x = Dense(64)(x)
		x = Dense(2)(x)  # represents mu, sigma
		
		#x = self.DistributionLayer(quantiles=quantiles)(x)  # returns 7 points, one for each quantile
		out = x
		
		model = Model(inputs=inp, outputs=out)
		
		return model
	
	# 2. trains a neural network (model) on the dataset
	def train_model(model,train_df,target_df):
		batch_size = 64
		epochs = 20
		num_classes = 7
		trained_model = model.fit(train_df, target_df, batch_size=batch_size,epochs=epochs,verbose=1)
	
	
		
		
	# 3. loads a test set
	def get_test_data(self):
		pass
	
	# 4. Uses the trained model to perform a prediction/classification
	def predict_output(self,train_df,target_df):
		
		pass
	
	# 1.1 Converts smaples/weights x into the 7 x-coordinates representing the 7 quantiles
	def DistributionLayer(self,quantiles,x):
		pass
	
if __name__ == '__main__':
	main= custom_layer_nn()
	train_df, target_df = main.get_train_data()
	
	quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
	inp_shape = [337,10]
	model = main.get_model(inp_shape, quantiles)
	print(f'model={model}')
	print("done")