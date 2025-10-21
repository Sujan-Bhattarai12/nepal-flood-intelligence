"""
LSTM Models for Flood Forecasting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


class FloodDataset(Dataset):
    """PyTorch Dataset for flood time series"""
    
    def __init__(self, data, lookback=168, forecast=168, stride=6):
        self.lookback_steps = lookback // stride
        self.forecast_steps = forecast // stride
        
        data = data.copy()
        data['discharge_cumecs'] = data['discharge_cumecs'].ffill().bfill()
        data['water_level_m'] = data['water_level_m'].ffill().bfill()
        
        self.discharge = data['discharge_cumecs'].values.astype(np.float32)
        self.water_level = data['water_level_m'].values.astype(np.float32)
        
        self.discharge_mean = self.discharge.mean()
        self.discharge_std = self.discharge.std()
        self.level_mean = self.water_level.mean()
        self.level_std = self.water_level.std()
        
        self.discharge_norm = (self.discharge - self.discharge_mean) / (self.discharge_std + 1e-8)
        self.level_norm = (self.water_level - self.level_mean) / (self.level_std + 1e-8)
        
        self.sequences = []
        self.targets = []
        
        for i in range(0, len(self.discharge) - self.lookback_steps - self.forecast_steps, stride):
            seq = np.stack([
                self.discharge_norm[i:i+self.lookback_steps],
                self.level_norm[i:i+self.lookback_steps]
            ], axis=-1)
            
            target = np.stack([
                self.discharge_norm[i+self.lookback_steps:i+self.lookback_steps+self.forecast_steps],
                self.level_norm[i+self.lookback_steps:i+self.lookback_steps+self.forecast_steps]
            ], axis=-1)
            
            self.sequences.append(seq)
            self.targets.append(target)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx]))
    
    def inverse_transform(self, normalized_data):
        """Convert normalized predictions back to original scale"""
        result = normalized_data.copy()
        result[..., 0] = normalized_data[..., 0] * self.discharge_std + self.discharge_mean
        result[..., 1] = normalized_data[..., 1] * self.level_std + self.level_mean
        return result


class LSTMModel(nn.Module):
    """LSTM architecture for flood forecasting"""
    
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=128, num_layers=2, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2 * 28)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.view(x.size(0), 28, 2)


class FloodForecaster:
    """Wrapper class for training and prediction"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.is_trained = False
        
    def train_silent(self, train_loader, val_loader, epochs=30):
        """Train model with early stopping"""
        best_val_loss = float('inf')
        patience = 0
        
        for epoch in range(epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = nn.MSELoss()(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    predictions = self.model(batch_x)
                    val_loss += nn.MSELoss()(predictions, batch_y).item()
            
            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    break
        
        self.is_trained = True
        return best_val_loss
    
    def predict(self, input_sequence, dataset=None):
        """Generate forecast from input sequence"""
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)
            prediction = self.model(input_tensor).cpu().numpy()[0]
        
        if dataset:
            prediction = dataset.inverse_transform(prediction)
        return prediction