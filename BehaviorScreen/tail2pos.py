import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

from BehaviorScreen.core import AGAROSE_WELL_DIMENSIONS
from BehaviorScreen.load import Directories, BehaviorData, load_data, find_files
from BehaviorScreen.process import get_circle_hough, get_background_image

## DATA ========================================================================

def get_features(behavior_data: BehaviorData):

    df = behavior_data.full_tracking
    pix_per_mm = behavior_data.metadata['calibration']['pix_per_mm']
    background_image = get_background_image(behavior_data)
    circle = get_circle_hough(
        background_image, pix_per_mm, 2, AGAROSE_WELL_DIMENSIONS, 2.5, 0.3
    )

    head = df.Head[['x', 'y']].to_numpy()
    sb = df.Swim_Bladder[['x', 'y']].to_numpy()
    heading_v = head - sb
    radial_v = sb - circle.center

    dot_w = np.einsum('ij,ij->i', heading_v, radial_v)
    det_w = heading_v[:, 0] * radial_v[:, 1] - heading_v[:, 1] * radial_v[:, 0]
    angle_to_wall = np.arctan2(det_w, dot_w)
    dist_from_center = np.linalg.norm(radial_v, axis=1)
    distance_to_wall = (circle.radius - dist_from_center) / circle.radius

    tail_coords = np.stack([df[f'Tail_{i}'][['x', 'y']].to_numpy() for i in range(10)], axis=1)
    segments = np.diff(tail_coords, axis=1) 
    h_expanded = -1*heading_v[:, np.newaxis, :]
    t_dot = h_expanded[..., 0] * segments[..., 0] + h_expanded[..., 1] * segments[..., 1]
    t_det = h_expanded[..., 0] * segments[..., 1] - h_expanded[..., 1] * segments[..., 0]
    tail_angles = np.arctan2(t_det, t_dot) 
    tail_velocity = np.diff(tail_angles, axis=0, prepend=tail_angles[:1, :])

    features = np.column_stack([
        tail_angles, 
        tail_velocity, 
        angle_to_wall, 
        distance_to_wall
    ])
    
    return features


def get_targets(behavior_data: BehaviorData):

    df = behavior_data.full_tracking
    h_x, h_y = df[('Head', 'x')].values, df[('Head', 'y')].values
    s_x, s_y = df[('Swim_Bladder', 'x')].values, df[('Swim_Bladder', 'y')].values
    
    theta = np.arctan2(h_y - s_y, h_x - s_x)
    diff_x = np.diff(h_x)
    diff_y = np.diff(h_y)
    diff_theta = np.diff(np.unwrap(theta))
    
    t_start = theta[:-1]
    dx_local = diff_x * np.cos(t_start) + diff_y * np.sin(t_start)
    dy_local = -diff_x * np.sin(t_start) + diff_y * np.cos(t_start)
    
    return np.column_stack([dx_local, dy_local, diff_theta])


def save_processed_data(behavior_file, save_dir, file_id):

    try:
        behavior_data = load_data(behavior_file)
        features = get_features(behavior_data) # (N, features)
        targets = get_targets(behavior_data)   # (N-1, 3)
        X_final = features[:-1, :]
        y_final = targets
        X_final = X_final.astype(np.float32)
        y_final = y_final.astype(np.float32)

        np.save(save_dir / f"X_{file_id}.npy", X_final)
        np.save(save_dir / f"y_{file_id}.npy", y_final)
        
        return X_final 
        
    except Exception as e:
        print(f"Error processing {behavior_file}: {e}")
        return None

def extract_data(
        base_path: Path,
        save_path: Path
    ):
    
    file_counter = 0
    all_features_for_scaling = []

    for line in base_path.iterdir():
        if not line.is_dir(): continue
        for condition in line.iterdir():
            if not condition.is_dir(): continue
            
            directories = Directories(
                root = condition,
                metadata='results',
                stimuli='results',
                tracking='results',
                full_tracking='lightning_pose',
                eyes_tracking='lightning_pose',
                temperature='results',
                video='results',
                video_timestamp='results',
                results='results',
                plots=''
            )

            for behavior_file in find_files(directories):
                print(f"Processing: {behavior_file.metadata}")
                X_data = save_processed_data(behavior_file, save_path, file_counter)

                if X_data is not None and file_counter % 10 == 0:
                    all_features_for_scaling.append(X_data[::10]) # Subsample further
                
                file_counter += 1

    print("Calculating global normalization...")
    full_sample = np.concatenate(all_features_for_scaling, axis=0)
    scaler = StandardScaler()
    scaler.fit(full_sample)
    joblib.dump(scaler, save_path / 'tcn_scaler.pkl')
    print(f"Done! Saved {file_counter} files and scaler to {save_path}")

## NETWORK ========================================================================

class FishSequenceDataset(Dataset):

    def __init__(
            self, 
            x_paths, 
            y_paths, 
            scaler, 
            window_size=30
        ):

        self.x_paths = sorted(x_paths) 
        self.y_paths = sorted(y_paths)
        self.scaler = scaler
        self.window_size = window_size
        
        self.lengths = [np.load(p, mmap_mode='r').shape[0] - window_size for p in x_paths]
        self.cumulative_lengths = np.cumsum(self.lengths)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        inner_idx = idx if file_idx == 0 else idx - self.cumulative_lengths[file_idx-1]
        
        # x_data = np.load(self.x_paths[file_idx], mmap_mode='r')
        # y_data = np.load(self.y_paths[file_idx], mmap_mode='r')
        x_data = np.load(self.x_paths[file_idx])
        y_data = np.load(self.y_paths[file_idx])

        x_raw = x_data[inner_idx : inner_idx + self.window_size]
        x_scaled = self.scaler.transform(x_raw) 
        y_val = y_data[inner_idx + self.window_size]
        
        return (torch.tensor(x_scaled, dtype=torch.float32).T, 
                torch.tensor(y_val, dtype=torch.float32))


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


## FUNCTIONS ========================================================================


def train(save_path: Path):

    x_files = sorted(list(save_path.glob("X_*.npy")))
    y_files = sorted(list(save_path.glob("y_*.npy")))

    scaler = joblib.load(save_path / 'tcn_scaler.pkl')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")


    model = FishTCN(input_size=20, output_size=3, num_channels=[64, 64, 64, 64]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    dataset = FishSequenceDataset(x_files, y_files, scaler, window_size=30)
    # increase num_workers if your CPU is fast, decrease if disk IO is the bottleneck
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), save_path / 'fish_tcn_model.pth')
    print("Model saved.")


def predict():
    ...

if __name__ == '__main__':
    
    BASE_PATH = Path('/media/martin/DATA_18TB/Screen')
    SAVE_PATH = Path('/media/martin/DATA_18TB/Processed_TCN_Data')
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    extract_data(BASE_PATH, SAVE_PATH)
    train(SAVE_PATH)