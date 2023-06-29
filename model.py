import torch
from torch import Tensor
import torch.nn as nn
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, out_channels, num_layers):
        super(CNN_LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers

        # Convolutional layer + pooling layer
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=self.out_channels, kernel_size=20, padding=0, stride=5)
            # nn.MaxPool1d(kernel_size=1, padding=0)
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

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# Transformer model
class Transformer(nn.Module):
    def __init__(self, feature_size, num_layers, nhead, d_model, dim_feedforward, enc_length, dropout):
        super(Transformer, self).__init__()

        # Global dropout used for all functions
        self.dropout = dropout

        # Encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                        dropout=self.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)

        # Decoder layer
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                        dropout=self.dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=num_layers)

        # Declares all masks
        self.memory_mask = None
        self.trg_mask = None

        # Value embedding through convolutional layers
        self.src_embedding = nn.Sequential(
            nn.Conv1d(in_channels=feature_size, out_channels=d_model, kernel_size=150, padding=0, stride=10),
            # nn.MaxPool1d(kernel_size=1, padding=0)
        )
        self.trg_embedding = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=1, padding='same'),
            # nn.MaxPool1d(kernel_size=1, padding=0)
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, self.dropout, enc_length)

        # Final linear layer
        self.linear_mapping = nn.Linear(in_features=d_model, out_features=1)

    def generate_square_subsequent_mask(self, dim1: int, dim2: int) -> Tensor:
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        Args:
            dim1: int, for both src and tgt masking, this must be target sequence
                  length
            dim2: int, for src masking this must be encoder sequence length (i.e.
                  the length of the input sequence to the model),
                  and for tgt masking, this must be target sequence length
        Return:
            A Tensor of shape [dim1, dim2]
        """
        return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

    def forward(self, src, trg, training=False):
        # Embed and encode encoder input
        src = self.src_embedding(src)
        src = src.permute(2, 0, 1)
        # shape ``[seq_len, batch_size, embedding_dim]``
        src = self.positional_encoding(src)

        # Embed and encode decoder input
        trg = trg.unsqueeze(1)
        trg = self.trg_embedding(trg)
        trg = trg.permute(2, 0, 1)
        trg = self.positional_encoding(trg)

        # Determines whether we need to apply masks to the decoder input
        if training:
            self.trg_mask = self.generate_square_subsequent_mask(dim1=trg.shape[0], dim2=trg.shape[0]).to(device)
            self.memory_mask = self.generate_square_subsequent_mask(dim1=trg.shape[0], dim2=src.shape[0]).to(device)
        else:
            self.trg_mask = None
            self.memory_mask = None

        memory = self.encoder(src)  # encoder
        output = self.decoder(trg, memory=memory, tgt_mask=self.trg_mask, memory_mask=self.memory_mask)  # decoder
        output = output.permute(1, 0, 2)
        output = self.linear_mapping(output)  # linear mapping layer

        return output
