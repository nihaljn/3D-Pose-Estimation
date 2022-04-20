import torch
import torch.nn as nn

class FrameModel(nn.Module):
    def __init__(self, n_joints, linear_size=1024):
        super().__init__()
        inp_size = 2*n_joints
        out_size = 3*n_joints
        self.n_joints = n_joints
        
        self.preprocessor = nn.Sequential(*[
            nn.Linear(inp_size, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        ])
        self.block1 = nn.Sequential(*[
            nn.Linear(linear_size, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(linear_size, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        ])
        self.block2 = nn.Sequential(*[
            nn.Linear(linear_size, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(linear_size, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        ])
        
        self.postprocessor = nn.Linear(linear_size, out_size)
        
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        y = self.preprocessor(x)
        y = self.block1(y) + y
        y = self.block2(y) + y
        y = self.postprocessor(y)
        y = y.view(x.shape[0], self.n_joints, 3)
        return y
    