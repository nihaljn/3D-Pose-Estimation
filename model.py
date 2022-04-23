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
    