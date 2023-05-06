import torch


def normalize(seq_data):
    # Compute mean and standard deviation for each column
    sequence_mean = seq_data.mean()
    sequence_std = seq_data.std()

    # Normalize the sequence data using vectorized operations
    sequence_norm = (seq_data - sequence_mean) / sequence_std

    return sequence_norm, sequence_mean, sequence_std


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
