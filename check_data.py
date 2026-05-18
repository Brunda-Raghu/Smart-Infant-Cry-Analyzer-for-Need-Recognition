#!/usr/bin/env python3
"""
Check if there are NaN or Inf values in the data or predictions
"""
import torch
import pickle

with open('train_cache_complete.pkl', 'rb') as f:
    X, y = pickle.load(f)

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"X min: {X.min():.4f}, max: {X.max():.4f}, mean: {X.mean():.4f}, std: {X.std():.4f}")
print(f"X has NaN: {torch.isnan(X).any()}")
print(f"X has Inf: {torch.isinf(X).any()}")
print(f"y unique values: {torch.unique(y)}")
print(f"y distribution: {[(i.item(), (y==i).sum().item()) for i in torch.unique(y)]}")
