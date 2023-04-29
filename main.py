import torch
import torch.nn as nn
import torch.optim as optim
from pkl_reader import readFile_sqlite
from model import CNN_LSTM
from utils import wrangle, prediction
from train import train_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""Define hyperparams"""
batch_size = 64
learning_rate = 0.001
epochs = 15
seq_length = 500
label_length = 100

# usable data:
# 9636/10
# 8101/31
"""Load data"""
data_path = "9636/9636_10.sqlite"
df = readFile_sqlite(data_path, transformation='simple')
train_dataset, train_dataloader, test_dataset, test_dataloader = wrangle(df, seq_length, label_length, batch_size)

"""Define optimizer and loss func"""
cnn_lstm = CNN_LSTM(df.shape[1], label_length).to(device)

optimizer = optim.Adam(cnn_lstm.parameters(), lr=learning_rate)
criterion = nn.L1Loss()

"""Train model"""
train_model(train_dataloader, test_dataloader, epochs, optimizer, criterion, cnn_lstm, device)
# Save model
torch.save(cnn_lstm.state_dict(), 'cnn_lstm.pth')

"""Get predictions"""
prediction(df, seq_length, label_length, cnn_lstm, device)
