import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from model import CNN_LSTM, Transformer, vLSTM
from utils import wrangle, graph_predictions, get_metrics, loadify, get_workload
from train import train_model
import neptune
import time


model_select = 'vLSTM'  # 'Transformer' or 'LSTM' or 'vLSTM'

# Neptune credentials
# model_n = neptune.init_model_version(
#     model="FOR-" + "LSTM",
#     project="nicogrima/Forecast-WL",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVu"
#               "ZS5haSIsImFwaV9rZXkiOiI0YjBhNDdmMS0zNDRkLTQ3YjctOTg4Yy0zODM3MGY5YmE2ZWEifQ=="
# )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if we want to train the model (keep as true unless you are using a saved one and do not want to train it further)
train_req = False
cross_val = False  # whether to use cross validation
single_file_load = True  # whether to load just one file
saved_model = True  # whether the model we are using has already been created
saved_mpath = model_select + '/vLSTM_300_v1.pth'

"""Define parameters"""
batch_size = 16
epochs = 2
feature_size = 65  # number of features
seq_length = 1200  # input length
label_length = 300
# Target we are looking to forecast (in the future we can modify the models to predict multiple)
label_target = 'Workload'
target_num = 64  # last column in df
learning_rate = 0.0001  # general learning rate

'''Define the model'''
if model_select == 'Transformer':
    learning_rate = 1e-7  # learning rate for specific model
    encoder_layers = 2
    decoder_layers = 2
    nhead = 1
    embed_dim = 1024
    dim_feedforward = 256
    dropout = 0.1
    e_ksize = 600  # cannot be greater than sequence length
    e_pad = 200
    e_stride = 30
    model = Transformer(feature_size, e_layers=encoder_layers, d_layers=decoder_layers, nhead=nhead, d_model=embed_dim,
                        dim_feedforward=dim_feedforward, dec_lenth=label_length, dropout=dropout,
                        e_kernel_size=e_ksize, e_padding=e_pad, e_stride=e_stride).to(device)
    params = {"learning_rate": learning_rate, "optimizer": "Adam", "loss_function": "MAE", "batch_size": batch_size,
              "epochs": epochs, "encoder_layers": encoder_layers, "decoder_layers": decoder_layers,"nheads": nhead,
              "embedding_dim": embed_dim, "dim_feedforward": dim_feedforward, "feature_size": feature_size,
              "sequence_length": seq_length, "label_length": label_length, "dropout": dropout,
              "kernel_size": e_ksize, "padding": e_pad, "stride": e_stride}

elif model_select == 'LSTM':
    learning_rate = 0.0005  # learning rate for specific model
    lstm_hidden_dim = 1028
    cnn_out_channels = 32
    lstm_layers = 1
    e_ksize = 50
    e_stride = 10
    e_pad = 20
    model = CNN_LSTM(feature_size, label_length, hidden_dim=lstm_hidden_dim, out_channels=cnn_out_channels,
                     num_layers=lstm_layers, kernel_size=e_ksize, stride=e_stride, padding=e_pad).to(device)
    params = {"learning_rate": learning_rate, "optimizer": "Adam", "loss_function": "MAE", "batch_size": batch_size,
              "epochs": epochs, "lstm_hidden_dim": lstm_hidden_dim, "cnn_out_channels": cnn_out_channels,
              "lstm_layers": lstm_layers, "feature_size": feature_size, "sequence_length": seq_length,
              "label_length": label_length, "kernel_size": e_ksize, "stride": e_stride, "padding": e_pad}

elif model_select == 'vLSTM':
    learning_rate = 0.0001  # learning rate for specific model
    lstm_hidden_dim = 512
    lstm_layers = 2
    model = vLSTM(feature_size, label_length, hidden_dim=lstm_hidden_dim, num_layers=lstm_layers).to(device)
else:
    raise ValueError('Model selection not available. Possible selections: "LSTM", "Transformer", or "vLSTM"')

# Set parameters for neptune
# model_n["parameters"] = params

if saved_model:
    model.load_state_dict(torch.load(saved_mpath))

"""Define optimizer and loss func"""
# Define optimizer as ADAM
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Define loss function as MAE/MSE
criterion = nn.L1Loss()
# StepLR decreases learning rate by gamma every step_size epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# usable data:
# 9636/10
# 8101/31
"""Load data"""
directory = 'C:/Users/nicos/PycharmProjects/NGC_data_compiler/Filled/'
if single_file_load:
    data_path = directory + 'f_9636_10_rel.csv'
    df = pd.read_csv(data_path)
    df = get_workload(df)  # adds workload level on the last column
    train_dataloader, test_dataloader = wrangle(df, seq_length, label_length, batch_size, label_target)
else:
    root = 'C:/Users/nicos/PycharmProjects/NGC_data_compiler/'
    train_data = np.load(root+'sequences_ab1.npz')
    train_array = train_data['arr']
    train_array = train_array[:, :, -seq_length:]
    train_array = np.transpose(train_array, (0, 2, 1))
    label_data = np.load(root+'labels_ab1.npz')
    label_array = label_data['arr']
    label_array = label_array[:, :label_length]
    train_dataloader, test_dataloader = loadify(train_array, label_array, batch_size)

"""Train model"""
save_file = model_select + '/' + model_select + '_f' + '.pth'  # model name + additional defining info + extension
# Train the model with either train-test split or k-folds cross-validation
if train_req:
    start_time = time.time()
    train_model(train_dataloader, test_dataloader, epochs, optimizer, criterion, model,
                    target_num, save_file, step=10, device=device, model_type=model_select, neptune=None,
                scheduler=scheduler)

    end_time = time.time()
    train_time = end_time - start_time
    print('Computation time for entire training: ' + str(train_time))
    # model_n["computation_time/training"] = round(train_time, 6)

"""Get prediction graph and metrics"""
if single_file_load:
    graph_predictions(df, seq_length, label_length, model, target_num, seq_length, device, model_select, None)

start_time = time.time()
get_metrics(test_dataloader, model, label_length, device, model_select, None)  # get performance metrics based on test data
end_time = time.time()
test_time = end_time-start_time
print('Computation time for entire testing set: ' + str(test_time))
