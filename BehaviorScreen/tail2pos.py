import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class FishSequenceDataset(Dataset):

    def __init__(
            self, 
            x_paths: list[Path], 
            y_paths: list[Path], 
            window_size: int = 30
        ):

        self.x_paths = x_paths 
        self.y_paths = y_paths
        self.window_size = window_size
        self.lengths = [np.load(p).shape[0] - window_size for p in x_paths]
        self.cumulative_lengths = np.cumsum(self.lengths)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        # Find which file the index belongs to
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        inner_idx = idx if file_idx == 0 else idx - self.cumulative_lengths[file_idx-1]
        
        # Load the file (In production, use memmap or cache the current file)
        x_data = np.load(self.x_paths[file_idx], mmap_mode='r')
        y_data = np.load(self.y_paths[file_idx], mmap_mode='r')
        
        # Extract window
        x_cond = x_data[inner_idx : inner_idx + self.window_size]
        y_cond = y_data[inner_idx + self.window_size]
        
        # TCN expects (Features, Time), so we transpose
        return torch.tensor(x_cond, dtype=torch.float32).T, torch.tensor(y_cond, dtype=torch.float32)

class ChanneledTCNBlock(nn.Module):

    def __init__(
            self, 
            n_inputs, 
            n_outputs, 
            kernel_size, 
            stride, 
            dilation, 
            padding, 
            dropout=0.2
        ):

        super(ChanneledTCNBlock, self).__init__()
        # Causal convolution: padding is (kernel_size-1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = nn.ConstantPad1d((-padding, 0), 0) # Remove "future" padding
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x))

class FishTCN(nn.Module):

    def __init__(
            self, 
            input_size, 
            output_size, 
            num_channels, 
            kernel_size=3, 
            dropout=0.2
        ):

        super(FishTCN, self). __init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [ChanneledTCNBlock(in_channels, out_channels, kernel_size, stride=1,
                                          dilation=dilation_size, padding=(kernel_size-1) * dilation_size, 
                                          dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x shape: (Batch, Features, Time) -> PyTorch Conv1d expects this
        y1 = self.network(x)
        # We only want the prediction for the LAST time step in the window
        return self.linear(y1[:, :, -1])

def train():

    # TODO: fill that it
    x_files = []
    y_files = []

    scaler = StandardScaler()
    model = FishTCN(input_size=20, output_size=3, num_channels=[64, 64, 64, 64])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    dataset = FishSequenceDataset(x_files, y_files, window_size=30)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    for epoch in range(10):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} complete.")

if __name__ == '__main__':
    
    train()