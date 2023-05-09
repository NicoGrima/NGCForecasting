import torch
import numpy as np
from utils import forecast, normalize
from model import CNN_LSTM, Transformer


'''Load the model parameters'''
# Load either the 'lstm' or 'transformer' model
model_to_load = 'transformer'

if model_to_load != 'lstm' and model_to_load != 'transformer':
    raise ValueError('Unavailable model selection. Available options: "lstm" or "transformer"')

filename = model_to_load + '.pth'
model = torch.load(filename, map_location=torch.device('cpu'))

'''Get your data'''
# Right now we get pseudo data here
target_num = 0  # whichever feature/column we want to predict
features = 119
input_length = 500
# input_tensor = torch.randn(1, featurs, input_length)
input_ndarray = np.random.normal(loc=50.0, scale=10.0, size=(1, features, input_length)).astype(np.float32)
norm_ndarray, seq_mean, seq_std = normalize(input_ndarray, target_num)
input_tensor = torch.from_numpy(norm_ndarray)
token_tensor = input_tensor[:, target_num, -1:]  # last element of target value in input tensor

'''Predict future data'''
norm_prediction = forecast(model, input_tensor, token_tensor, model_to_load)
norm_prediction = norm_prediction.detach().to('cpu').numpy().squeeze()
prediction = (norm_prediction * seq_std[target_num]) + seq_mean[target_num]
# Now transform continuous estimate to ordinal classification
bins = [0, 33, 67, 100]  # define the bin edges
labels = ['underload', 'optimal', 'overload']  # define the corresponding labels for the bins
# bin the values and map the bin indices to the labels
classified_prediction = np.array(labels)[np.digitize(prediction, bins) - 1]
