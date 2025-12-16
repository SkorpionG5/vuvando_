import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
import wfdb
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
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

def preprocess_patient_data(patient_id, data_dir='mitdb'):

    if not os.path.exists(data_dir):
        wfdb.dl_database('mitdb', data_dir)

    record = wfdb.rdrecord(os.path.join(data_dir, patient_id))
    annotation = wfdb.rdann(os.path.join(data_dir, patient_id), 'atr')

    window_size = 187
    half_window = window_size // 2

    signal = record.p_signal[:, 0] 
    peaks = annotation.sample
    labels = annotation.symbol

    normal_beats = []
    abnormal_beats = []

    for i, (peak, label) in enumerate(zip(peaks, labels)):
        if peak < half_window or peak > len(signal) - half_window:
            continue
        
        beat = signal[peak - half_window : peak + half_window + 1]
        
        if len(beat) != window_size:
            beat = np.resize(beat, window_size)

        if label == 'N':
            normal_beats.append(beat)
        elif label == 'V': 
            abnormal_beats.append(beat)

    X_normal = np.array(normal_beats)
    X_abnormal = np.array(abnormal_beats)

    if len(X_normal) > 0:
        scaler = MinMaxScaler()
        X_normal = scaler.fit_transform(X_normal)
        if len(X_abnormal) > 0:
            X_abnormal = scaler.transform(X_abnormal)
    
    return X_normal, X_abnormal

def load_data(partition_id: int, num_partitions: int):

    patient_ids = ['100', '101', '102', '103', '104', '105', '106', '108', '112', '113']
    
    patient_id = patient_ids[partition_id % len(patient_ids)]
    
    print(f"Client {partition_id} loading Patient {patient_id}...")

    try:
        X_normal, X_abnormal = preprocess_patient_data(patient_id)
        
        if len(X_normal) == 0:
            raise ValueError(f"Patient {patient_id} has no Normal beats!")

        tensor_train = torch.Tensor(X_normal)
        tensor_test = torch.Tensor(X_normal[:50]) 

        if len(X_abnormal) > 0:
            tensor_abnormal = torch.Tensor(X_abnormal)
            tensor_test = torch.cat((tensor_test, tensor_abnormal), 0)

        trainloader = DataLoader(TensorDataset(tensor_train), batch_size=32, shuffle=True)
        testloader = DataLoader(TensorDataset(tensor_test), batch_size=32)
        
        return trainloader, testloader

    except Exception as e:
        print(f"Error loading Patient {patient_id}: {e}")
        return DataLoader(TensorDataset(torch.randn(10, 187)), batch_size=32), \
               DataLoader(TensorDataset(torch.randn(10, 187)), batch_size=32)

def get_proximal_loss(net, global_params, lambda_reg):

    proximal_term = 0.0
    for local_param, global_param in zip(net.parameters(), global_params):
        proximal_term += (local_param - global_param).norm(2) ** 2
    
    return (lambda_reg / 2) * proximal_term

def train(net, trainloader, epochs, lr, device, global_params=None, lambda_reg=1.0):

    net.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    if global_params:
        global_params = [p.to(device) for p in global_params]

    net.train()
    running_loss = 0.0
    
    for _ in range(epochs):
        for batch in trainloader:
            signals = batch[0].to(device)
            optimizer.zero_grad()

            outputs = net(signals)
            loss_recon = criterion(outputs, signals)

            loss_proximal = 0.0
            if global_params is not None:
                loss_proximal = get_proximal_loss(net, global_params, lambda_reg)

            loss = loss_recon + loss_proximal
            
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
