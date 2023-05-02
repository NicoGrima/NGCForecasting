import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, out_channels, num_layers):
        super(CNN_LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers

        # Convolutional layer + pooling layer
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=self.out_channels, kernel_size=1, padding='same'),
            nn.MaxPool1d(kernel_size=1, padding=0)
        )

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.out_channels, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                            batch_first=True)

        # Final linear layer
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=output_dim)

    def forward(self, x):
        # Convolutional layers
        x = self.cnn(x)

        # LSTM layer
        x = x.permute(0, 2, 1)  # change dimensions for LSTM input (batch_size, seq_length, input_size)
        # Initialize empty hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Linear layer
        out = self.fc(out[:, -1, :])
        return out


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# Transformer model
class Transformer(nn.Module):
    def __init__(self, feature_size, num_layers, nhead, d_model, dim_feedforward, enc_length):
        super(Transformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

        self.src_embedding = nn.Conv1d(feature_size, d_model, kernel_size=1, padding='same')
        self.trg_embedding = nn.Conv1d(1, d_model, kernel_size=1, padding='same')

        self.positional_encoding = PositionalEncoding(d_model, 0.1, enc_length)

        self.linear = nn.Linear(d_model, 1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, trg, training=False):
        if training:
            self.trg_mask = self.generate_square_subsequent_mask(trg.shape[1]).to(device)
        else:
            self.trg_mask = None

        src = self.src_embedding(src)
        src = src.permute(2, 0, 1)
        src = self.positional_encoding(src)

        trg = trg.unsqueeze(1)
        trg = self.trg_embedding(trg)
        trg = trg.permute(2, 0, 1)
        trg = self.positional_encoding(trg)

        memory = self.encoder(src, mask=self.src_mask)
        output = self.decoder(trg, memory, tgt_mask=self.trg_mask, memory_mask=self.memory_mask)
        output = output.permute(1, 0, 2)
        output = self.linear(output)

        return output
