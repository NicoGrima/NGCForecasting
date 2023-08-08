import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import time


def normalize(seq_data, label_target='target', label=0):
    # Whether to discard the sequence (if values are stale/unchanging)
    discard = False

    # Compute mean and standard deviation for each column
    sequence_mean = seq_data.mean()
    sequence_std = seq_data.std()

    # Handle floating point precision error
    if (sequence_std < 1e-4).any() and sequence_std[label_target] < 1e-1:
        discard = True
        print('Discarded: low overall std')
    elif (sequence_std < 1e-4).any():
        discard = True
        print('Discarded: low input std')
    elif sequence_std[label_target] < 1e-1:
        discard = True
        print('Discarded: low label std')

    # Normalize the sequence data using vectorized operations
    sequence_norm = (seq_data - sequence_mean) / sequence_std

    # Normalize the label based on the mean and standard deviation of the target value computed previously
    target_num = seq_data.columns.get_loc(label_target)
    label_norm = (label - sequence_mean[target_num]) / sequence_std[target_num]

    return sequence_norm.values, label_norm, sequence_mean.values, sequence_std.values, discard


def create_sequences(input_data: pd.DataFrame, label_target: str,
                     sequence_length=1000, label_length=100, step=1):
    sequences = []
    labels = []
    data_size = len(input_data)

    # Normalize
    # for column in input_data.columns:
    #     _mean = mean[input_data.columns.get_loc(column)]
    #     _std = std[input_data.columns.get_loc(column)]
    #     input_data[column] = (input_data[column] - _mean) / _std

    for i in range(0, data_size - sequence_length - label_length, step):
        # Define the range for the sequence
        sequence = input_data[i:i + sequence_length]

        label_start = i + sequence_length
        label_end = label_start + label_length
        # Define the label range and target features
        label = input_data[label_target][label_start:label_end]

        # Append values to list
        # sequences.append(sequence)
        # labels.append(label)

        # # Normalize the sequences
        sequence_norm, label_norm, _, _, discard = normalize(sequence, label_target, label)
        sequence_norm = np.transpose(sequence_norm, (1, 0))

        sequences.append(sequence_norm)
        labels.append(label_norm)

        # seqs_nan = np.any(np.isnan(sequence_norm))
        # label_nan = np.any(np.isnan(label_norm))
        # what = np.isnan(sequence_norm)
        # op = np.argwhere(what)
        # if seqs_nan or label_nan:
        #     print('found')

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)


def loadify(train_array, label_array, batch_size):
    # print(np.any(np.isnan(train_array)))
    # print(np.any(np.isnan(label_array)))

    train_data, test_data, train_labels, test_labels = train_test_split(train_array, label_array, test_size=0.2,
                                                                        random_state=42)

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


def wrangle(df, seq_length, label_length, batch_size, label_target='Workload'):
    # From full data analysis
    # normal_values = pd.read_csv('C:/Users/nicos/PycharmProjects/NGC_data_compiler/normal_vals.csv')

    # Normalize the values
    # for column in df.columns:
    #     df[column] = (df[column] - normal_values[column][0]) / normal_values[column][1]

    input_data, labels = create_sequences(df, label_target, seq_length, label_length)

    # Train-test split
    train_data, test_data, train_labels, test_labels = train_test_split(input_data, labels, test_size=0.2,
                                                                        random_state=42)

    # Convert train and test data and labels to PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    # Create PyTorch DataLoader objects for train and test data
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def get_workload(df: pd.DataFrame):
    workload = 0 * df['prob_Underload'] + 50 * df['prob_Optimal'] + 100 * df['prob_Overload']
    # plt.plot(workload)
    # plt.show()
    df.drop(['workload_classification', 'prob_Underload', 'prob_Optimal', 'prob_Overload'], axis=1,
            inplace=True)
    df['Workload'] = workload

    return df


def get_metrics(test_dataloader, model, label_length, device, model_type: str, neptune):
    real_vals = []
    predicted_vals = []

    # Get predicted outcomes for every testing sequence and compare results to the labels
    with torch.no_grad():
        if model_type == 'LSTM' or model_type == 'vLSTM':
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
                token_input = enc_input[:, :, -label_length:]
                dec_input = token_input
                # target_num = 64
                # step = 10
                # token_input = enc_input[:, target_num, -step:]
                # token_input = token_input.unsqueeze(1)
                # dec_input = token_input

                # Generate the output sequence iteratively
                outputs = model.forward(enc_input, dec_input)
                # for t in range(labels.size(1)):
                #     outputs = model.forward(enc_input, dec_input, training=False)
                #     dec_input = torch.cat((token_input, outputs.squeeze(2)), dim=1)
                # outputs = None
                # for t in range(0, labels.size(1), step):
                #     outputs = model.forward(enc_input, dec_input)
                #     # Add output (t+1 prediction) to the decoder input, then repeat
                #     out = outputs.permute(0, 2, 1)
                #     dec_input = torch.cat((dec_input, out[:, :, -step:]), dim=2)
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

    metrics = {"mae": mae, "mse": mse}
    # neptune["eval"] = metrics

    # Accuracy for predictions
    # accuracy = acct / (acct + false)
    # print('Accuracy for entire prediction of the time series is: ' + str(accuracy))


def graph_predictions(df, seq_length, label_length, model, target_num, enc_length, device, model_type: str,
                      neptune):
    # Initialize predicted with the first sequence of length = seq_length (input length), since we
    # cannot predict these values
    predicted = df.iloc[:, target_num][0:seq_length].to_numpy()
    target_name = df.columns[target_num]
    comp_times = []
    model.eval()

    if model_type == 'LSTM' or model_type == 'vLSTM':
        for t in range(seq_length, df.shape[0] - label_length + 1, label_length):
            data = df[t - seq_length:t]  # input data
            # Normalize the data and store the mean and standard deviation
            norm_data, _, norm_mean, norm_std, discard = normalize(data, target_name)
            if discard:
                continue
            norm_tensor = torch.tensor([norm_data], dtype=torch.float32).to(device)  # normalized input
            norm_tensor = norm_tensor.permute(0, 2, 1)
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
            norm_data, _, norm_mean, norm_std, discard = normalize(data, target_name)
            if discard:
                continue
            norm_tensor = torch.tensor([norm_data], dtype=torch.float32)  # normalized input
            norm_tensor = norm_tensor.permute(0, 2, 1)
            token_tensor = norm_tensor[:, :, -label_length:]  # decoder input
            # token_input = norm_tensor[:, target_num, -10:]
            # token_input = token_input.unsqueeze(1)
            dec_tensor = token_tensor.to(device)
            input_tensor = norm_tensor.to(device)
            start_time = time.time()  # start time
            # outputs = None
            # for t in range(0, label_length, 10):
            #     outputs = model.forward(input_tensor, dec_tensor)
            #     # Add output (t+1 prediction) to the decoder input, then repeat
            #     out = outputs.permute(0, 2, 1)
            #     dec_tensor = torch.cat((dec_tensor, out[:, :, -10:]), dim=2)
            outputs = model.forward(input_tensor, dec_tensor)
            prediction = outputs.cpu().detach().numpy()  # forecasted target values
            # Inverse normalization of predicted values
            prediction = list(map(lambda x: (prediction * norm_std[target_num] + norm_mean[target_num]), prediction))
            end_time = time.time()  # end time
            predicted = np.append(predicted, prediction[0][0])
            comp_times.append(end_time - start_time)  # store computation time

    # Account for sequence length mismatch with total data size
    leftover = (df.shape[0] % label_length) * -1 if df.shape[0] % label_length != 0 else df.shape[0]

    # Computation time average
    print('Time of computation for each prediction is: ' + str(np.mean(comp_times)))
    # neptune["computation_time/single"] = round(float(np.mean(comp_times)), 6)

    """Plot predicted vs real"""
    time_range = range(len(predicted))
    time_range = time_range[enc_length:]
    real = df.iloc[:, target_num].to_numpy()[:leftover]  # trim real values if needed

    plt.plot(time_range, real[enc_length:], label='True')
    plt.plot(time_range, predicted[enc_length:], label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Workload')
    plt.title('Forecasting Cognitive Workload')
    plt.legend()
    plt.show()

    real = pd.DataFrame(real[enc_length:], columns=['Real Workload'])
    predicted = pd.DataFrame(predicted[enc_length:], columns=['Predicted Workload'])
    final_df = pd.concat([real, predicted], axis=1)
    final_df.to_csv('conti_predict.csv', index=False)

    def convert_to_cat(n):
        if n <= 33:
            return 'low'
        elif n < 67:
            return 'medium'
        else:
            return 'high'

    final_df = final_df.applymap(convert_to_cat)
    final_df.to_csv('cat_predict.csv', index=False)

    acc = 0

    for index, row in final_df.iterrows():
        if row['Real Workload'] == row['Predicted Workload']:
            acc += 1

    total = final_df.shape[0]
    accuracy = (acc/total) * 100
    accuracy = round(accuracy, 2)

    print('Accuracy of predictions: ' + str(accuracy) + '%')

    # Compute confusion matrix
    true = final_df['Real Workload']
    pred = final_df['Predicted Workload']
    cm = confusion_matrix(true, pred, labels=['low', 'medium', 'high'])

    # Print the confusion matrix
    print(cm)

    # Optionally, to visualize it, use seaborn heatmap
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    ax.set_xticklabels(['Low', 'Medium', 'High'])
    ax.set_yticklabels(['Low', 'Medium', 'High'])
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    plt.show()


