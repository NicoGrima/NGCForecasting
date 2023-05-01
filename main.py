import torch
import torch.nn as nn
import torch.optim as optim
from pkl_reader import readFile_sqlite
from model import CNN_LSTM
from utils import wrangle, graph_predictions, wrangle_cross, get_metrics
from train import train_model, cross_train_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cross_val = False

"""Define hyperparams"""
batch_size = 64
learning_rate = 0.001
epochs = 25
seq_length = 500
label_length = 100

# usable data:
# 9636/10
# 8101/31
"""Load data"""
data_path = "9636/9636_10.sqlite"
df = readFile_sqlite(data_path, transformation='simple')

if not cross_val:
    train_dataloader, test_dataloader = wrangle(df, seq_length, label_length, batch_size)
else:
    fold_dataloader, test_dataloader = wrangle_cross(df, seq_length, label_length, batch_size, k_folds=5)

"""Define optimizer and loss func"""
cnn_lstm = CNN_LSTM(df.shape[1], label_length).to(device)

optimizer = optim.Adam(cnn_lstm.parameters(), lr=learning_rate)
criterion = nn.L1Loss()

"""Train model"""
if not cross_val:
    train_model(train_dataloader, test_dataloader, epochs, optimizer, criterion, cnn_lstm, device)
else:
    cross_train_model(fold_dataloader, epochs, optimizer, criterion, cnn_lstm, device)

"""Get prediction graph and metrics"""
graph_predictions(df, seq_length, label_length, cnn_lstm, device)
get_metrics(test_dataloader, cnn_lstm, device)
