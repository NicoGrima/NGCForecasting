import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time


def normalize(seq_data, label_target='target', label=0):
    # Compute mean and standard deviation for each column
    sequence_mean = seq_data.mean()
    sequence_std = seq_data.std()

    # Normalize the sequence data using vectorized operations
    sequence_norm = (seq_data - sequence_mean) / sequence_std

    # Normalize the label based on the mean and standard deviation of the target value computed previously
    target_num = seq_data.columns.get_loc(label_target)
    label_norm = (label - sequence_mean[target_num]) / sequence_std[target_num]

    return sequence_norm.values.transpose(), label_norm, sequence_mean.values, sequence_std.values


def create_sequences(input_data: pd.DataFrame, label_target: str, sequence_length=10, label_length=1):
    sequences = []
    labels = []
    data_size = len(input_data)

    for i in range(data_size - sequence_length - label_length):
        # Define the range for the sequence
        sequence = input_data[i:i + sequence_length]

        label_start = i + sequence_length
        label_end = label_start + label_length
        # Define the label range and target features
        label = input_data[label_target][label_start:label_end]

        # Normalize the sequences
        sequence_norm, label_norm, i, o = normalize(sequence, label_target, label)
        sequences.append(sequence_norm)
        labels.append(label_norm)

    return np.array(sequences), np.array(labels)


def loadify(train_array, label_array, batch_size):
    # print(np.any(np.isnan(train_array)))
    # print(np.any(np.isnan(label_array)))

    train_data, test_data, train_labels, test_labels = train_test_split(train_array, label_array, test_size=0.2)

    # Convert train and test data and labels to PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    # Create PyTorch DataLoader objects for train and test data
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def wrangle(df, seq_length, label_length, batch_size, label_target='Workload', cross_val='False', k_folds=5,
            load=False):
    if not load:
        input_data, labels = create_sequences(df, label_target, seq_length, label_length)
    # else:
    #     input_data = train_array
    #     labels = label_array

    # Train-test split
    train_data, test_data, train_labels, test_labels = train_test_split(input_data, labels, test_size=0.2)

    # Convert train and test data and labels to PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    # Create PyTorch DataLoader objects for train and test data
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if cross_val:
        # If k-fold cross-validation is chosen
        kfold = KFold(n_splits=k_folds, shuffle=True)
        fold_dataloaders = []

        # Iterate through each of the k-subsets and create a DataLoader for each
        for train_indices, val_indices in kfold.split(train_dataset):
            train_subset = Subset(train_dataset, train_indices)
            val_subset = Subset(train_dataset, val_indices)

            train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            fold_dataloaders.append((train_dataloader, val_dataloader))

        return fold_dataloaders, test_dataloader

    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_dataloader, test_dataloader


def get_metrics(test_dataloader, model, target_num, device, model_type: str):
    real_vals = []
    predicted_vals = []

    # Get predicted outcomes for every testing sequence and compare results to the labels
    with torch.no_grad():
        if model_type == 'LSTM':
            # Iterate through all testing sequences
            for i, (inputs, labels) in enumerate(test_dataloader):
                inputs = inputs.to(device)
                labels = np.array(labels)
                real_vals.append(labels)

                # Get predicted outputs and store them for comparison with real labels
                outputs = np.array(model.forward(inputs).to('cpu'))
                predicted_vals.append(outputs)

        elif model_type == 'Transformer':
            for i, (enc_input, labels) in enumerate(test_dataloader):
                enc_input = enc_input.to(device)
                real_vals.append(np.array(labels))
                labels = labels.to(device)

                # Initialize the decoder input with a start token (here, we use the first target value)
                token_input = enc_input[:, target_num, -1:]
                dec_input = token_input

                # Generate the output sequence iteratively
                outputs = None
                for t in range(labels.size(1)):
                    outputs = model.forward(enc_input, dec_input, training=False)
                    dec_input = torch.cat((token_input, outputs.squeeze(2)), dim=1)
                outputs = np.array(outputs.to('cpu'))
                predicted_vals.append(np.squeeze(outputs))

    # Transform labels and predicted outcomes into desired data format
    real_vals = np.vstack(real_vals)
    predicted_vals = np.vstack(predicted_vals)
    print(real_vals.shape)

    # MAE for predictions
    mae = mean_absolute_error(predicted_vals, real_vals)
    bv = np.mean(np.abs(predicted_vals - real_vals), axis=1)
    hw = np.mean(bv)

    print('MAE for entire predictions of the time series is: ' + str(mae))

    # MSE for predictions
    mse = mean_squared_error(predicted_vals, real_vals)
    print('MSE for entire predictions of the time series is: ' + str(mse))

    # Accuracy for predictions
    # accuracy = acct / (acct + false)
    # print('Accuracy for entire prediction of the time series is: ' + str(accuracy))


def graph_predictions(df, seq_length, label_length, model, target_num, enc_length, device, model_type: str):
    # Initialize predicted with the first sequence of length = seq_length (input length), since we
    # cannot predict these values
    predicted = df.iloc[:, target_num][0:seq_length].to_numpy()
    target_name = df.columns[target_num]
    comp_times = []

    if model_type == 'LSTM':
        for t in range(seq_length, df.shape[0] - label_length + 1, label_length):
            data = df[t - seq_length:t]  # input data
            # Normalize the data and store the mean and standard deviation
            norm_data, _, norm_mean, norm_std = normalize(data, target_name)
            norm_tensor = torch.tensor([norm_data], dtype=torch.float32).to(device)  # normalized input
            start_time = time.time()  # start time
            prediction = model.forward(norm_tensor).cpu().detach().numpy()  # forecasted target values
            # Inverse normalization of predicted values
            prediction = list(map(lambda x: (prediction * norm_std[target_num] + norm_mean[target_num]), prediction))
            end_time = time.time()  # end time
            predicted = np.append(predicted, prediction[0][0])
            comp_times.append(end_time - start_time)  # store computation time

    elif model_type == 'Transformer':
        for t in range(enc_length, df.shape[0] - label_length + 1, label_length):
            data = df[t - enc_length:t]  # input data
            # Normalize the data and store the mean and standard deviation
            norm_data, _, norm_mean, norm_std = normalize(data, target_name)
            norm_tensor = torch.tensor([norm_data], dtype=torch.float32)  # normalized input
            token_tensor = norm_tensor[:, target_num, -1].unsqueeze(-1).to(device)  # decoder input
            dec_tensor = token_tensor
            input_tensor = norm_tensor.to(device)
            start_time = time.time()  # start time
            prediction = []
            output = None
            # Iterate through model to predict t+1 and progressively update the outcome and decoder input
            # with predictions
            for i in range(label_length):
                output = model.forward(input_tensor, dec_tensor, training=False)
                dec_tensor = torch.cat((token_tensor, output.squeeze(2)), dim=1)  # update decoder input
            prediction = output.cpu().detach().numpy()  # forecasted target values
            # Inverse normalization of predicted values
            prediction = list(map(lambda x: (prediction * norm_std[target_num] + norm_mean[target_num]), prediction))
            end_time = time.time()  # end time
            predicted = np.append(predicted, prediction[0][0])
            comp_times.append(end_time - start_time)  # store computation time

    # Account for sequence length mismatch with total data size
    leftover = (df.shape[0] % label_length) * -1 if df.shape[0] % label_length != 0 else df.shape[0]

    # Computation time average
    print('Time of computation for each prediction is: ' + str(np.mean(comp_times)))

    """Plot predicted vs real"""
    time_range = range(len(predicted))
    real = df.iloc[:, target_num].to_numpy()[:leftover]  # trim real values if needed

    plt.plot(time_range, real, label='True')
    plt.plot(time_range, predicted, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Workload')
    plt.title('Forecasting Cognitive Workload')
    plt.legend()
    plt.show()
