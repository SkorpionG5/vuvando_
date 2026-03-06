"""vuvando: A Flower / PyTorch app for Anomaly Detection (Thesis)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
import wfdb
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------------------------------------
# 1. THE MODEL (Autoencoder)
# -----------------------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input: 187 samples (approx 0.5s of ECG at 360Hz)
        self.encoder = nn.Sequential(
            nn.Linear(187, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 187),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# -----------------------------------------------------------------------------
# 2. DATA PROCESSING (The Hard Part - Handling MIT-BIH)
# -----------------------------------------------------------------------------
def preprocess_patient_data(patient_id, data_dir='mitdb'):
    """
    Reads a specific patient's record, segments heartbeats, and normalizes them.
    """
    # Download dataset if not exists
    if not os.path.exists(data_dir):
        wfdb.dl_database('mitdb', data_dir)

    # Read Signal and Annotations
    record = wfdb.rdrecord(os.path.join(data_dir, patient_id))
    annotation = wfdb.rdann(os.path.join(data_dir, patient_id), 'atr')

    # MIT-BIH is sampled at 360Hz. 
    # We take 187 samples per beat (approx 0.5 seconds), centered on the peak.
    window_size = 187
    half_window = window_size // 2

    signal = record.p_signal[:, 0] # Use Lead I (Index 0)
    peaks = annotation.sample
    labels = annotation.symbol

    normal_beats = []
    abnormal_beats = []

    # Segment beats based on annotations
    for i, (peak, label) in enumerate(zip(peaks, labels)):
        # Skip beats too close to start or end
        if peak < half_window or peak > len(signal) - half_window:
            continue
        
        # Cut the window
        beat = signal[peak - half_window : peak + half_window + 1]
        
        # Ensure exact length (sometimes off by 1)
        if len(beat) != window_size:
            beat = np.resize(beat, window_size)

        # Separate Normal (N) vs Abnormal (V) for Anomaly Detection
        if label == 'N':
            normal_beats.append(beat)
        elif label == 'V': # Premature Ventricular Contraction
            abnormal_beats.append(beat)

    # Convert to Numpy
    X_normal = np.array(normal_beats)
    X_abnormal = np.array(abnormal_beats)

    # Normalize to [0, 1] range (Crucial for Neural Networks)
    # Note: We fit scaler only on Normal data to simulate "Normalcy"
    if len(X_normal) > 0:
        scaler = MinMaxScaler()
        X_normal = scaler.fit_transform(X_normal)
        if len(X_abnormal) > 0:
            X_abnormal = scaler.transform(X_abnormal)
    
    return X_normal, X_abnormal

def load_data(partition_id: int, num_partitions: int):
    """
    Load real MIT-BIH data for a specific client.
    """
    # MIT-BIH has these patients (we select a subset for simulation)
    # We map 'partition_id' (0, 1, 2...) to real Patient IDs (100, 101...)
    patient_ids = ['100', '101', '102', '103', '104', '105', '106', '108', '112', '113']
    
    # Safety check: Wrap around if we ask for Client 11 but only have 10 patients
    patient_id = patient_ids[partition_id % len(patient_ids)]
    
    print(f"Client {partition_id} loading Patient {patient_id}...")

    try:
        X_normal, X_abnormal = preprocess_patient_data(patient_id)
        
        # THESIS STRATEGY:
        # Train: Only on Normal beats (The model learns "Normality")
        # Test: On Normal + Abnormal (To check if it reconstructs well)
        
        if len(X_normal) == 0:
            raise ValueError(f"Patient {patient_id} has no Normal beats!")

        # Convert to PyTorch Tensors
        tensor_train = torch.Tensor(X_normal)
        tensor_test = torch.Tensor(X_normal[:50]) # Use subset of normal for test
        
        # If patient has abnormal beats, add them to test set
        if len(X_abnormal) > 0:
            tensor_abnormal = torch.Tensor(X_abnormal)
            tensor_test = torch.cat((tensor_test, tensor_abnormal), 0)

        trainloader = DataLoader(TensorDataset(tensor_train), batch_size=32, shuffle=True)
        testloader = DataLoader(TensorDataset(tensor_test), batch_size=32)
        
        return trainloader, testloader

    except Exception as e:
        print(f"Error loading Patient {patient_id}: {e}")
        # Fallback to dummy data if download fails (prevents crash)
        return DataLoader(TensorDataset(torch.randn(10, 187)), batch_size=32), \
               DataLoader(TensorDataset(torch.randn(10, 187)), batch_size=32)


# -----------------------------------------------------------------------------
# 3. TRAINING LOOPS (Same as before)
# -----------------------------------------------------------------------------
def train(net, trainloader, epochs, lr, device):
    net.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            signals = batch[0].to(device)
            optimizer.zero_grad()
            outputs = net(signals)
            loss = criterion(outputs, signals)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return running_loss / len(trainloader)

def test(net, testloader, device):
    net.to(device)
    net.eval()
    criterion = nn.MSELoss()
    loss = 0.0
    with torch.no_grad():
        for batch in testloader:
            signals = batch[0].to(device)
            outputs = net(signals)
            loss += criterion(outputs, signals).item()
    return loss / len(testloader), 0.0
