#!/usr/bin/env python3
"""
Debug the model input/output shapes
"""
import torch
import torch.nn as nn
import pickle

class OldDeepInfantModel(nn.Module):
    """Original model architecture used in predict.py"""
    def __init__(self, num_classes=9):
        super(OldDeepInfantModel, self).__init__()
        
        # CNN layers (c)
        self.c = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        
        # LSTM layers (l)
        self.l = nn.LSTM(
            input_size=256*10,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Classifier (f)
        self.f = nn.Linear(512, num_classes)  # 512 = 256*2 (bidirectional)
    
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        
        # CNN processing
        x = self.c(x)
        print(f"After CNN: {x.shape}")  # (batch, 256, freq', time')
        
        # Permute to (batch, time, channels, freq)
        x = x.permute(0, 3, 1, 2)
        print(f"After permute: {x.shape}")
        
        b, t, c, f = x.size()
        print(f"Dimensions: batch={b}, time={t}, channels={c}, freq={f}")
        
        # Reshape to (batch, time, features)
        x = x.reshape(b, t, c * f)
        print(f"After reshape: {x.shape}")
        
        # LSTM
        x, _ = self.l(x)
        print(f"After LSTM: {x.shape}")
        
        # Take last timestep
        x = x[:, -1, :]
        print(f"After last timestep: {x.shape}")
        
        # Classifier
        x = self.f(x)
        print(f"After classifier: {x.shape}")
        
        return x

# Load data
with open('train_cache_complete.pkl', 'rb') as f:
    X, y = pickle.load(f)

print(f"Data shape: X={X.shape}, y={y.shape}")

# Test model
model = OldDeepInfantModel()
with torch.no_grad():
    output = model(X[:1])
    print(f"Output: {output}")
