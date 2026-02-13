import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import os
import joblib

# Configuration
DATA_FILE = "v2/dataset/prometheus_baseline.csv"
MODEL_PATH = "v2/ml/lstm_model.pth"
SCALER_PATH = "v2/ml/scaler.pkl"
THRESHOLD_PATH = "v2/ml/threshold.txt"
SEQUENCE_LENGTH = 10
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 32  # Increased capacity slightly
LATENT_DIM = 8     # Bottleneck size

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=8):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden2latent = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Encoder Pass
        _, (hidden, _) = self.encoder(x) # hidden: (1, batch, hidden_dim)
        
        # Bottleneck (Compression)
        # Squeeze to (batch, hidden) -> Linear -> (batch, latent)
        latent = self.hidden2latent(hidden[-1])
        
        # Expand for Decoder (Repeat vector for each time step)
        # (batch, latent) -> (batch, hidden) -> (batch, seq_len, hidden)
        hidden_decoded = self.latent2hidden(latent)
        hidden_repeated = hidden_decoded.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decoder Pass
        decoded, _ = self.decoder(hidden_repeated)
        
        # Reconstruction
        output = self.output_layer(decoded)
        return output

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

def augment_data(data, noise_level=0.01):
    """Add Gaussian noise to avoid overfitting to perfect 'idle' lines"""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def train():
    print(f"Loading data from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        print("Data file not found!")
        return

    df = pd.read_csv(DATA_FILE)
    features = ['cpu', 'mem', 'net_in', 'net_out']
    data = df[features].values
    
    # Normalize
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    joblib.dump(scaler, SCALER_PATH)
    
    # Create sequences
    X = create_sequences(data_scaled, SEQUENCE_LENGTH)
    
    # Data Augmentation (Robustness)
    X_train = augment_data(X)
    
    # Convert to Tensors
    X_tensor = torch.FloatTensor(X_train)
    dataset = TensorDataset(X_tensor, X_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model Setup
    model = LSTMAutoencoder(input_dim=len(features), hidden_dim=HIDDEN_SIZE, latent_dim=LATENT_DIM)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting training on {len(X)} sequences...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_in, batch_target in loader:
            optimizer.zero_grad()
            output = model(batch_in)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(loader):.6f}")
            
    # Save Model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # --- CALCULATE THRESHOLD ---
    print("Calculating anomaly threshold...")
    model.eval()
    with torch.no_grad():
        # Use original (non-augmented) data for thresholding
        original_tensor = torch.FloatTensor(X)
        reconstructions = model(original_tensor)
        
        # Calculate MAE per sequence (Mean over features and time)
        # Shape: (batch, seq, feat)
        loss = torch.mean(torch.abs(reconstructions - original_tensor), dim=(1,2))
        loss_values = loss.numpy()
        
        # 3-Sigma Rule
        mean_loss = np.mean(loss_values)
        std_loss = np.std(loss_values)
        threshold = mean_loss + 3 * std_loss
        
        print(f"Reconstruction Error - Mean: {mean_loss:.4f}, Std: {std_loss:.4f}")
        print(f"Selected Threshold (Mean + 3*Std): {threshold:.4f}")
        
        with open(THRESHOLD_PATH, "w") as f:
            f.write(str(threshold))
        print(f"Threshold saved to {THRESHOLD_PATH}")

if __name__ == "__main__":
    train()
