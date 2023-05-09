import torch
import numpy as np
import os
import sys
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, base_path)
from utils import forecast, normalize, format_data
from model import CNN_LSTM, Transformer

class UFForcaster():
    """Class for providing forcasted predictions of workload states"""
    def __init__(self, model_type):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load either the 'lstm' or 'transformer' model
        self.model_to_load = model_type
        if self.model_to_load != 'lstm' and self.model_to_load != 'transformer':
            raise ValueError('Unavailable model selection. Available options: "lstm" or "transformer"')
        filename = self.model_to_load + '.pth'
        file_path = os.path.join(base_path, filename)
        self.model = torch.load(file_path, map_location=self.device).to(self.device)

        # Define whichever feature/column we want to predict
        self.target_num = 0

    def predict(self, light_data, processed_data, classified_data):
        '''Predict forecast based on current data'''
        # Convert data into expected format
        data = format_data(
            light_data,
            processed_data,
            classified_data
        )
        data = data.astype(np.float32)

        # Convert data into input tensor
        target_num = 0  
        norm_ndarray, seq_mean, seq_std = normalize(data)
        norm_ndarray = np.reshape(norm_ndarray, [1, len(data.columns), len(data)])
        input_tensor = torch.from_numpy(norm_ndarray).to(self.device)
        token_tensor = input_tensor[:, self.target_num, -1:].to(self.device)  # last element of target value in input tensor

        # Get prediction
        norm_prediction = forecast(self.model, input_tensor, token_tensor, self.model_to_load)
        norm_prediction = norm_prediction.detach().to('cpu').numpy().squeeze()
        prediction = (norm_prediction * seq_std[target_num]) + seq_mean[target_num]  

        # Now transform continuous estimate to ordinal classification
        bins = [0, 33, 67, 100]  # define the bin edges
        labels = ['Underload', 'Optimal', 'Overload']  # define the corresponding labels for the bins
        # bin the values and map the bin indices to the labels
        classified_prediction = np.array(labels)[np.digitize(prediction, bins) - 1]
        return classified_prediction
