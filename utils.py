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
    sequence_norm = []
    sequence_mean = []
    sequence_std = []
    target_num = seq_data.columns.get_loc(label_target)
    for col in seq_data.columns:
        mean = seq_data[col].mean()
        sequence_mean.append(mean)
        std = seq_data[col].std()
        sequence_std.append(std)
        norm_seq = (seq_data[col]-mean)/std
        sequence_norm.append(np.array(norm_seq))
    label_norm = (label - sequence_mean[target_num]) / sequence_std[target_num]
    return np.array(sequence_norm), label_norm, sequence_mean, sequence_std


def create_sequences(input_data: pd.DataFrame, label_target: str, sequence_length=10, label_length=1):
    sequences = []
    labels = []
    data_size = len(input_data)

    for i in range(data_size - sequence_length - label_length):
        sequence = input_data[i:i + sequence_length]  #.to_numpy()

        label_start = i + sequence_length
        label_end = label_start + label_length
        label = input_data[label_target][label_start:label_end]

        sequence_norm, label_norm, _, _ = normalize(sequence, label_target, label)
        sequences.append(sequence_norm)
        labels.append(label_norm)

    return np.array(sequences), np.array(labels)


def wrangle(df, seq_length, label_length, batch_size, label_target='Workload', cross_val='False', k_folds=5):
    input_data, labels = create_sequences(df, label_target, seq_length, label_length)

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
        # Perform k-fold cross-validation
        kfold = KFold(n_splits=k_folds, shuffle=True)
        fold_dataloaders = []

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

    with torch.no_grad():
        if model_type == 'LSTM':
            for i, (inputs, labels) in enumerate(test_dataloader):
                inputs = inputs.to(device)
                labels = np.array(labels)
                real_vals.append(labels)

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
                predicted_vals.append(outputs)

    real_vals = np.vstack(real_vals)
    predicted_vals = np.vstack(predicted_vals)
    print(real_vals.shape)

    # MAE for predictions
    mae = mean_absolute_error(predicted_vals, real_vals)
    print('MAE for entire predictions of the time series is: ' + str(mae))

    # MSE for predictions
    mse = mean_squared_error(predicted_vals, real_vals)
    print('MSE for entire predictions of the time series is: ' + str(mse))

    # Accuracy for predictions
    # accuracy = acct / (acct + false)
    # print('Accuracy for entire prediction of the time series is: ' + str(accuracy))


def graph_predictions(df, seq_length, label_length, model, device, model_type: str):
    predicted = df['Workload'][0:seq_length].to_numpy()
    comp_times = []

    for t in range(seq_length, df.shape[0] - label_length + 1, label_length):
        data = df[t - seq_length:t]
        norm_data, _, norm_mean, norm_std = normalize(data.to_numpy())
        norm_tensor = torch.tensor([norm_data], dtype=torch.float32).to(device)
        start_time = time.time()  # Start time
        prediction = model.forward(norm_tensor).cpu().detach().numpy()
        prediction = list(map(lambda x: (prediction * norm_std[2] + norm_mean[2]), prediction))
        end_time = time.time()  # End time
        predicted = np.append(predicted, prediction[0][0])
        comp_times.append(end_time - start_time)

    leftover = (df.shape[0] % label_length) * -1 if df.shape[0] % label_length != 0 else df.shape[0]

    # Computation time average
    print('Time of computation for each prediction is: ' + str(np.mean(comp_times)))

    """Plot predicted vs real"""
    days = range(len(predicted))
    real = df['Workload'].to_numpy()[:leftover]

    # Plot the data
    plt.plot(days, real, label='True')
    plt.plot(days, predicted, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Workload')
    plt.title('Forecasting Cognitive Workload')
    plt.legend()
    plt.show()
