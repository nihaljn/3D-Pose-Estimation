import math
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, dropout=0.5, linear_size=1024):
        super().__init__()
        self.block = nn.Sequential(*[
            nn.Linear(linear_size, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(linear_size, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        ])
        
    def forward(self, x):
        return x + self.block(x)


class FrameModel(nn.Module):
    def __init__(self, n_joints, linear_size=1024, dropout=0.5, n_blocks=2):
        super().__init__()
        inp_size = 2*n_joints
        out_size = 3*n_joints
        self.n_joints = n_joints
        self.preprocessor = nn.Sequential(*[
            nn.Linear(inp_size, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        ])
        self.blocks = nn.Sequential(*[Block(dropout=dropout, linear_size=linear_size) for _ in range(n_blocks)])
        self.postprocessor = nn.Linear(linear_size, out_size)
        
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        y = self.preprocessor(x)
        y = self.blocks(y)
        y = self.postprocessor(y)
        y = y.view(x.shape[0], self.n_joints, 3)
        return y
    

class SequenceModel(nn.Module):
    def __init__(self, n_joints, linear_size, n_encoder_heads=3, n_encoder_layers=3, dropout=0.5):
        super().__init__()
        inp_size = 2*n_joints
        out_size = 3*n_joints
        self.n_joints = n_joints
        self.linear_size = linear_size
        self.preprocessor = nn.Sequential(*[
            nn.Linear(inp_size, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        ])
        self.pos_encoder = PositionalEncoding(linear_size, dropout=0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=linear_size, nhead=n_encoder_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.decoder = nn.Sequential(*[
            nn.Linear(linear_size, linear_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(linear_size, out_size)
        ])
        
    def forward(self, x, src_mask=None):
        assert len(x.shape) == 4
        assert x.shape[0] == 1 # x should be of shape (1, T, n_joints, 2)
        x = x.view(x.shape[1], -1)
        x = self.preprocessor(x)
        x = x.unsqueeze(0) # x is of shape (1, T, linear_size)
        # x = x * math.sqrt(self.linear_size)
        # x = self.pos_encoder(x)
        x = self.encoder(x, mask=src_mask)
        x = x.squeeze() # x is of shape (T, linear_size)
        y = self.decoder(x)
        y = y.view(x.shape[0], self.n_joints, 3)
        return y
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''x: Tensor, shape [batch_size, seq_len, embedding_dim]'''
        x = x + self.pe[:, :x.size(0)]
        return self.dropout(x)