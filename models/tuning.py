from numpy import einsum
import torch
import torch.nn as nn
import torch.nn.functional as F



class DLinear(nn.Module):
    def __init__(self, configs):
        super(DLinear, self).__init__()
        input_len = configs.seq_len
        output_len = configs.pred_len
        self.hidden_dm = 128
        self.linear1 = nn.Linear(input_len, self.hidden_dm)
        self.linear2 = nn.Linear(self.hidden_dm, output_len)
        self.activation = nn.LeakyReLU()
        

    def forward(self, x):
        # x: [Batch, Input length,Channel]

        x = self.linear1(x.permute(0,2,1))
        x = self.activation(x)
        x = self.linear2(x).permute(0,2,1)
        return x # to [Batch, Output length, Channel]


class SimpleFCNet(nn.Module):
    def __init__(self, configs, sequence_length, in_features):
        super(SimpleFCNet, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.weight0 = nn.Parameter(torch.randn(sequence_length, in_features), requires_grad=True)
        self.weight1 = nn.Parameter(F.softplus(torch.randn(sequence_length, in_features)), requires_grad=True)
        self.weight2 = nn.Parameter(F.softplus(torch.randn(sequence_length, in_features)), requires_grad=True)
        self.fc1 = nn.Linear(in_features*sequence_length, 2096)
        self.fc2 = nn.Linear(2096, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 2096)
        self.fc6 = nn.Linear(2096, in_features*sequence_length)
        self.activation = nn.Tanh()

    def forward(self, x):
        
        batch_size = x.size(0)
        sequence_length = x.size(1)
        x_flattened = x.reshape(batch_size, -1)  
        
        x = nn.Tanh()(self.fc1(x_flattened))
        x = nn.Tanh()(self.fc2(x))
        x = nn.Tanh()(self.fc3(x))
        x = nn.Tanh()(self.fc4(x))
        x = nn.Tanh()(self.fc5(x))
        x = self.fc6(x)
        x = x.unsqueeze(1)

        x_reshaped = x.reshape(batch_size, sequence_length, -1)  
        
        return x_reshaped

class SimpleFCNet2(nn.Module):
    def __init__(self, configs, sequence_length, in_features):
        super(SimpleFCNet2, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.hidden_dm = 1024

        self.fc1 = nn.Linear(in_features*sequence_length, self.hidden_dm)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(self.hidden_dm, in_features*sequence_length)
        self.activation = nn.ReLU()

    def forward(self, x):

        # Normalization
        # means = x.mean(1, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x = x / stdev
        # print(x.shape)
        batch_size = x.size(0)
        sequence_length = x.size(1)
        x_flattened = x.reshape(batch_size, -1)  
        
        x = self.activation(self.fc1(x_flattened))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)

        x = x.unsqueeze(1)

        x_reshaped = x.reshape(batch_size, sequence_length, -1) 

        # Applying the inverse transformations
        # x_reshaped = x_reshaped * stdev + means 
        
        return x_reshaped
    

class SimpleFCNet_traffic(nn.Module):
    def __init__(self, configs, sequence_length):
        super(SimpleFCNet_traffic, self).__init__()
        self.configs = configs

        self.hidden_dm = 128
        self.drop1 = nn.Dropout(0.4)
        self.drop2 = nn.Dropout(0.4)
        self.norm = nn.LayerNorm(sequence_length)
        self.fc1 = nn.Linear(sequence_length, self.hidden_dm)
        self.fc2 = nn.Linear(self.hidden_dm, 256)
        self.fc3 = nn.Linear(256, self.hidden_dm)
        self.fc4 = nn.Linear(self.hidden_dm, sequence_length)
        self.activation = nn.GELU()

    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(1)
        dim = x.size(2)

        x = x.reshape(batch_size,sequence_length*dim)

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        v = self.norm(x)
        x = self.activation(self.fc1(x))
        x = self.drop1(x)
        x = self.activation(self.fc2(x))
        x = self.drop2(x)
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        x = x*v

        x = x * stdev + means

        x = x.squeeze(dim=-1)
        x_reshaped = x.view(batch_size, sequence_length, dim)  
        
        return x_reshaped    

class SimpleFCNet_channel(nn.Module):
    def __init__(self, configs, sequence_length):
        super(SimpleFCNet_channel, self).__init__()
        self.configs = configs
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.hidden_dm = 256
        self.dim = configs.cseg_len

        self.fc1 = nn.Linear(sequence_length, self.hidden_dm*self.dim)
        self.fc2 = nn.Linear(self.hidden_dm*self.dim, sequence_length)
        self.activation = nn.GELU()
    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(1)
        feature = x.size(2)
        # dim = x.size(2)

        x = x.reshape(batch_size, sequence_length*self.dim)

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        x = x * stdev + means
        x = x.squeeze(dim=1)
        x_reshaped = x.view(batch_size, sequence_length, self.dim)  
        
        return x_reshaped

class SimpleFCNet_timestep(nn.Module):
    def __init__(self, configs, sequence_length):
        super(SimpleFCNet_timestep, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.hidden_dm = 256
        self.dim = configs.cseg_len

        self.fc1 = nn.Linear(sequence_length, self.hidden_dm*self.dim)
        self.fc2 = nn.Linear(self.hidden_dm*self.dim, sequence_length)
        self.activation = nn.GELU()
    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(1)
        feature = x.size(2)
        # dim = x.size(2)

        x = x.reshape(batch_size, feature*self.dim)

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        x = x * stdev + means
        x = x.squeeze(dim=1)
        x_reshaped = x.view(batch_size, sequence_length, feature)  
        
        return x_reshaped



class SimpleFCNet_ett(nn.Module):
    def __init__(self, configs, sequence_length):
        super(SimpleFCNet_ett, self).__init__()
        self.configs = configs

        self.hidden_dm = 512
        self.drop1 = nn.Dropout(0.4)
        self.drop2 = nn.Dropout(0.4)
        self.norm = nn.LayerNorm(sequence_length)
        self.fc1 = nn.Linear(sequence_length, self.hidden_dm)
        self.fc2 = nn.Linear(self.hidden_dm, 128)
        self.fc3 = nn.Linear(128, self.hidden_dm)
        self.fc4 = nn.Linear(self.hidden_dm, sequence_length)
        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(1)
        dim = x.size(2)

        x = x.reshape(batch_size,sequence_length*dim)

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        v = self.norm(x)
        x = self.activation(self.fc1(x))
        x = self.drop1(x)
        x = self.activation(self.fc2(x))
        x = self.drop2(x)
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        x = x*v

        x = x * stdev + means

        x = x.squeeze(dim=-1)
        x_reshaped = x.view(batch_size, sequence_length, dim)  
        
        return x_reshaped  

class SimpleFCNet_exchange(nn.Module):
    def __init__(self, configs, sequence_length):
        super(SimpleFCNet_exchange, self).__init__()
        self.configs = configs

        self.hidden_dm = 256
        self.norm = nn.LayerNorm(sequence_length)
        self.fc1 = nn.Linear(sequence_length, self.hidden_dm)
        self.fc2 = nn.Linear(self.hidden_dm, 128)
        self.fc3 = nn.Linear(128, self.hidden_dm)
        self.fc4 = nn.Linear(self.hidden_dm, sequence_length)
        self.activation = nn.ReLU()
    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(1)
        # dim = x.size(2)
        dim = 1
        x = x.reshape(batch_size,sequence_length*dim)

        x = self.activation(self.fc1(x))
        # x = self.activation(self.fc2(x))
        # x = self.activation(self.fc3(x))
        x = self.fc4(x)

        # x = x * stdev + means

        x = x.squeeze(dim=-1)
        x_reshaped = x.view(batch_size, sequence_length, dim)  
        
        return x_reshaped  