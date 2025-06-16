from numpy import einsum
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFCNet_exchange(nn.Module):
    def __init__(self, configs, sequence_length):
        super(SimpleFCNet_exchange, self).__init__()
        self.configs = configs

        self.hidden_dm = 256
        self.norm = nn.LayerNorm(sequence_length)
        self.fc1 = nn.Linear(sequence_length, self.hidden_dm)
        self.fc2 = nn.Linear(self.hidden_dm, sequence_length)
        self.activation = nn.ReLU()
    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(1)
        dim = x.size(2)
        # dim = 1
        x = x.reshape(batch_size,sequence_length*dim)

        x = self.activation(self.fc1(x))

        x = self.fc2(x)

        x = x.squeeze(dim=-1)
        x_reshaped = x.view(batch_size, sequence_length, dim)  
        
        return x_reshaped  