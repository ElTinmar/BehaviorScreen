from pathlib import Path
import joblib
import random

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    
    behavior_files = [
        bf
        for line in base_path.iterdir() if line.is_dir()
        for condition in line.iterdir() if condition.is_dir()
        for bf in find_files(Directories(
            root=condition,
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
        ))
    ]

    random.shuffle(behavior_files)
    train_files, val_files = train_test_split(
        behavior_files,
        test_size=0.2
    )

    train_scaling_samples = []
    for split, files in [("train", train_files), ("val", val_files)]:
        print(f"=== Processing {split} ===")
        for idx, behavior_file in enumerate(files):
            X_data = save_processed_data(behavior_file, save_path, f"{split}_{idx}")
            if split == "train" and X_data is not None and idx % 10 == 0:
                train_scaling_samples.append(X_data[::10])

    print("Calculating normalization (train only)...")
    train_sample = np.concatenate(train_scaling_samples, axis=0)
    scaler = StandardScaler()
    scaler.fit(train_sample)
    joblib.dump(scaler, save_path / 'tcn_scaler.pkl')

## NETWORK ========================================================================

class FishSequenceDataset(Dataset):

    def __init__(
            self, 
            x_paths, 
            y_paths, 
            scaler, 
            window_size=30,
        ):

        assert len(x_paths) == len(y_paths)

        self.x_paths = sorted(x_paths) 
        self.y_paths = sorted(y_paths)
        self.scaler = scaler
        self.window_size = window_size
        self.current_file_idx = -1

        # Load everythin to RAM
        print('loading data to RAM...')
        self.x_data = {}
        self.y_data = {}
        self.lengths = []
        pairs = list(zip(self.x_paths, self.y_paths))
        for file_idx, (xp, yp) in enumerate(tqdm(pairs)):
            x_raw = np.load(xp)
            self.x_data[file_idx] = torch.from_numpy(self.scaler.transform(x_raw)).float()
            self.y_data[file_idx] = torch.from_numpy(np.load(yp)).float()
            self.lengths.append(self.x_data[file_idx].shape[0] - window_size)
        
        self.cumulative_lengths = np.cumsum(self.lengths)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        inner_idx = idx if file_idx == 0 else idx - self.cumulative_lengths[file_idx-1] 
        x = self.x_data[file_idx][inner_idx : inner_idx + self.window_size]
        y = self.y_data[file_idx][inner_idx + self.window_size]
        return x.T, y


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
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = nn.ConstantPad1d((-padding, 0), 0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        
        # Residual mapping if input/output channels differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res) 


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

def validate(
        model, 
        loader, 
        criterion, 
        device,
        max_batches=50
    ):
    
    model.eval()
    val_loss = 0
    steps = 0 
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(loader):
            if i >= max_batches: 
                break
            steps += 1
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            val_loss += loss.item()
    
    avg_loss = val_loss / steps
    model.train() 
    return avg_loss
    
def train(
        save_path: Path, 
        validate_every: int = 500,
        n_workers: int = 4,
        batch_size: int = 512
    ):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train = sorted(list(save_path.glob("X_train_*.npy")))
    y_train = sorted(list(save_path.glob("y_train_*.npy")))
    x_val = sorted(list(save_path.glob("X_val_*.npy")))
    y_val = sorted(list(save_path.glob("y_val_*.npy")))
    x_scaler = joblib.load(save_path / 'tcn_scaler.pkl')

    train_ds = FishSequenceDataset(x_train, y_train, x_scaler)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    val_ds = FishSequenceDataset(x_val, y_val, x_scaler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)

    model = FishTCN(input_size=20, output_size=3, num_channels=[64, 64, 128, 128]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    writer = SummaryWriter(log_dir=save_path / "logs")
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(10):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/10", unit="batch")
        epoch_loss = 0
        
        model.train()
        for i, (batch_x, batch_y) in enumerate(pbar):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss
            if i % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{current_loss:.6f}",
                    "avg_loss": f"{epoch_loss/(i+1):.6f}"
                })

            writer.add_scalar("Loss/Train", current_loss, global_step)
            global_step += 1
            
            if global_step % validate_every == 0:
                current_val_loss = validate(model, val_loader, criterion, device)
                writer.add_scalar("Loss/Validation", current_val_loss, global_step)
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    torch.save(model.state_dict(), save_path / 'best_model.pth')

        print(f"--- Epoch {epoch+1} Finished. Avg Loss: {epoch_loss/len(train_loader):.6f} ---")

    writer.close()

def predict(model, dataset, file_idx, device):
    model.eval()
    
    # 1. Prepare data for a single file
    # We want to predict the entire file, so we iterate through its specific indices
    start_idx = 0 if file_idx == 0 else dataset.cumulative_lengths[file_idx-1]
    end_idx = dataset.cumulative_lengths[file_idx]
    
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for i in range(start_idx, end_idx):
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device) # Add batch dimension (1, Features, Window)
            
            output = model(x)
            predictions.append(output.squeeze().cpu().numpy())
            ground_truth.append(y.numpy())
            
    predictions = np.array(predictions) # (N, 3) -> [dx, dy, dtheta]
    ground_truth = np.array(ground_truth)
    
    # 2. Reconstruct Global Trajectory
    def reconstruct(moves):
        # We start at origin (0,0) with 0 heading
        x, y, theta = 0, 0, 0
        traj = [[x, y]]
        
        for dx_loc, dy_loc, d_theta in moves:
            # Update heading
            theta += d_theta
            # Rotate local move back to global coordinates
            dx_glob = dx_loc * np.cos(theta) - dy_loc * np.sin(theta)
            dy_glob = dx_loc * np.sin(theta) + dy_loc * np.cos(theta)
            
            x += dx_glob
            y += dy_glob
            traj.append([x, y])
        return np.array(traj)

    traj_pred = reconstruct(predictions)
    traj_real = reconstruct(ground_truth)
    
    # 3. Plotting
    plt.figure(figsize=(10, 5))
    
    # Subplot 1: XY Trajectory
    plt.subplot(1, 2, 1)
    plt.plot(traj_real[:, 0], traj_real[:, 1], 'k-', alpha=0.5, label='Actual')
    plt.plot(traj_pred[:, 0], traj_pred[:, 1], 'r--', alpha=0.8, label='Predicted')
    plt.title("Reconstructed Trajectory")
    plt.xlabel("X (local units)")
    plt.ylabel("Y (local units)")
    plt.legend()
    plt.axis('equal')
    
    # Subplot 2: Heading change comparison
    plt.subplot(1, 2, 2)
    plt.plot(ground_truth[:200, 2], 'k', label='Actual dTheta')
    plt.plot(predictions[:200, 2], 'r--', label='Predicted dTheta')
    plt.title("Heading Change (First 200 frames)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    return predictions, ground_truth

if __name__ == '__main__':
    
    BASE_PATH = Path('/media/martin/DATA_18TB/Screen')
    SAVE_PATH = Path('/home/martin/Documents/Processed_TCN_Data')
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    #extract_data(BASE_PATH, SAVE_PATH)
    train(SAVE_PATH)