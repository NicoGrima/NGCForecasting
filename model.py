import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN_LSTM, self).__init__()

        self.hidden_dim = 64
        self.out_channels = 32
        self.num_layers = 1

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=self.out_channels, kernel_size=1, padding='same'),
            nn.MaxPool1d(kernel_size=1, padding=0)
        )

        self.lstm = nn.LSTM(input_size=self.out_channels, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True)

        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=output_dim)

    def forward(self, x):
        # Convolutional layers
        x = self.cnn(x)

        # LSTM layer
        x = x.permute(0, 2, 1)  # Change dimensions for LSTM input (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Fully connected layer
        out = self.fc(out[:, -1, :])
        return out