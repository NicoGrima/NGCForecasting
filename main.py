import torch
import numpy as np
from utils import forecast
from model import CNN_LSTM, Transformer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''Load the model parameters'''
# Load either the 'lstm' or 'transformer' model
model_to_load = 'transformer'

if model_to_load != 'lstm' and model_to_load != 'transformer':
    raise ValueError('Unavailable model selection. Available options: "lstm" or "transformer"')

filename = model_to_load + '.pth'
model = torch.load(filename).to(device)

'''Get your data'''
# Right now we get pseudo data here
target_num = 0  # whichever feature/column we want to predict
features = 119
input_length = 500
input_tensor = torch.randn(1, features, input_length).to(device)
token_tensor = input_tensor[:, target_num, -1:].to(device)  # last element of target value in input tensor

'''Predict future data'''
prediction = forecast(model, input_tensor, token_tensor, model_to_load)
prediction = prediction.detach().to('cpu').numpy().squeeze()
# Now transform continuous estimate to ordinal classification
bins = [0, 33, 67, 100]  # define the bin edges
labels = ['underload', 'optimal', 'overload']  # define the corresponding labels for the bins
# bin the values and map the bin indices to the labels
classified_prediction = np.array(labels)[np.digitize(prediction, bins) - 1]
