import torch


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
