import torch
import numpy as np


def normalize(seq_data, target_num=0):
    seq_data = seq_data.squeeze()
    sequence_norm = []
    sequence_mean = []
    sequence_std = []
    # Iterate through each feature and normalize them based on the feature values in the sequence
    for col in seq_data:
        mean = np.mean(col)
        sequence_mean.append(mean)
        std = np.std(col)
        sequence_std.append(std)
        norm_seq = (col-mean)/std  # computation for z-score normalization
        sequence_norm.append(np.array(norm_seq))
    return np.expand_dims(np.array(sequence_norm), axis=0), np.array(sequence_mean), np.array(sequence_std)


def forecast(model, input_tensor, token_tensor, model_to_load):
    output = None
    if model_to_load == 'lstm':
        output = model.forward(input_tensor)
    elif model_to_load == 'transformer':
        label_length = 100  # update based on desired/trained label length
        dec_tensor = token_tensor
        for prediction in range(label_length):
            output = model.forward(input_tensor, dec_tensor)
            dec_tensor = torch.cat((token_tensor, output.squeeze(2)), dim=1)
    else:
        raise ValueError('No such model is available')

    return output
