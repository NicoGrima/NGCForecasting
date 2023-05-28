import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from pkl_reader import readFile_sqlite
from model import CNN_LSTM, Transformer
from utils import wrangle, graph_predictions, get_metrics, loadify
from train import train_model, cross_train_model
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if we want to train the model (keep as true unless you are using a saved one and do not want to train it further)
train_req = True
cross_val = False  # whether to use cross validation
single_file_load = True  # whether to load just one file
saved_model = False  # whether the model we are using has already been created
model_select = 'LSTM'  # 'Transformer' or 'LSTM'

"""Define parameters"""
batch_size = 64
epochs = 5
feature_size = 65  # number of features
seq_length = 700  # input length
label_length = 100
# Target we are looking to forecast (in the future we can modify the models to predict multiple)
label_target = 'Workload'
target_num = 0
learning_rate = 0.0001  # general learning rate

'''Define the model'''
if model_select == 'Transformer':
    learning_rate = 0.000001  # learning rate for specific model
    model = Transformer(feature_size, num_layers=6, nhead=8, d_model=64,
                        dim_feedforward=64, enc_length=seq_length).to(device)
elif model_select == 'LSTM':
    learning_rate = 0.001  # learning rate for specific model
    model = CNN_LSTM(feature_size, label_length, hidden_dim=64, out_channels=32,
                     num_layers=1).to(device)
else:
    raise ValueError('Model selection not available. Possible selections: "LSTM" or "Transformer"')

if saved_model:
    filename = 'cnn_lstm_30.pth'
    model.load_state_dict(torch.load(filename))

"""Define optimizer and loss func"""
# Define optimizer as ADAM
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Define loss function as MAE
criterion = nn.L1Loss()

# usable data:
# 9636/10
# 8101/31
"""Load data"""
if single_file_load:
    data_path = "9636/9636_10.sqlite"  #  "9636/9636_10.sqlite"
    # Read and manipulate data from sqlite
    df = readFile_sqlite(data_path, transformation='simple')
    # list of x most important features
    # features = ['Workload', 'opt8_HbO', 'opt12_HbO', 'opt10_HbT', 'opt12_HbR', 'opt12_Oxy', 'opt1_HbR', 'opt2_Oxy', 'opt10_HbO', 'opt14_HbT', 'opt5_HbR']
    # df = df[features]
    train_dataloader, test_dataloader = wrangle(df, seq_length, label_length, batch_size,
                                                label_target, cross_val=cross_val, k_folds=5)
else:
    train_data = np.load('C:/Users/nicos/PycharmProjects/NGC_data_compiler/sequences.npz')
    train_array = train_data['arr']
    train_array = np.transpose(train_array, (0, 2, 1))
    label_data = np.load('C:/Users/nicos/PycharmProjects/NGC_data_compiler/labels.npz')
    label_array = label_data['arr']
    train_dataloader, test_dataloader = loadify(train_array, label_array, batch_size)

"""Train model"""
save_file = model_select + '_mock' + '.pth'  # model name + additional defining info + extension
# Train the model with either train-test split or k-folds cross-validation
if train_req:
    start_time = time.time()
    if not cross_val:
        train_model(train_dataloader, test_dataloader, epochs, optimizer, criterion, model,
                    target_num, save_file, device, model_select)
    else:
        cross_train_model(train_dataloader, epochs, optimizer, criterion, model, target_num,
                          save_file, device, model_select)
    end_time = time.time()
    train_time = end_time - start_time
    print('Computation time for entire training: ' + str(train_time))

"""Get prediction graph and metrics"""
if single_file_load:
    graph_predictions(df, seq_length, label_length, model, target_num, seq_length, device, model_select)

start_time = time.time()
get_metrics(test_dataloader, model, target_num, device, model_select)  # get performance metrics based on test data
end_time = time.time()
test_time = end_time-start_time
print('Computation time for entire testing set: ' + str(test_time))
