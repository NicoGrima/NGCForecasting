import torch
import torch.nn as nn
import torch.optim as optim
from pkl_reader import readFile_sqlite
from model import CNN_LSTM, Transformer
from utils import wrangle, graph_predictions, get_metrics
from train import train_model, cross_train_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cross_val = False  # whether to use cross validation
model_select = 'Transformer'  # 'Transformer' or 'LSTM'

# usable data:
# 9636/10
# 8101/31
"""Load data"""
data_path = "9636/9636_10.sqlite"
df = readFile_sqlite(data_path, transformation='simple')
# df = df.iloc[:1000]

"""Define hyperparams"""
batch_size = 64
epochs = 15
feature_size = 119
seq_length = 500
label_length = 100
label_target = 'Workload'
target_num = df.columns.get_loc(label_target)
if model_select == 'Transformer':
    learning_rate = 0.00001
    model = Transformer(feature_size, num_layers=6, nhead=8, d_model=32,
                        dim_feedforward=64, enc_length=seq_length).to(device)
elif model_select == 'LSTM':
    learning_rate = 0.001
    model = CNN_LSTM(feature_size, label_length).to(device)
else:
    raise ValueError('Model selection not available. Possible selections: "LSTM" or "Transformer"')

"""Define optimizer and loss func"""
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.L1Loss()

"""Train model"""
train_dataloader, test_dataloader = wrangle(df, seq_length, label_length, batch_size,
                                            label_target, cross_val=cross_val, k_folds=5)

if not cross_val:
    train_model(train_dataloader, test_dataloader, epochs, optimizer, criterion, model,
                target_num, device, model_select)
else:
    cross_train_model(train_dataloader, epochs, optimizer, criterion, model, target_num,
                      device, model_select)

"""Get prediction graph and metrics"""
graph_predictions(df, seq_length, label_length, model, target_num, seq_length, device, model_select)
get_metrics(test_dataloader, model, target_num, device, model_select)
