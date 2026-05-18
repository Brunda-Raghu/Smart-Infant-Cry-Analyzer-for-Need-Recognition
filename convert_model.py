import torch
from train import DeepInfantModel

# Load the old model checkpoint
old_checkpoint = torch.load('deepinfant.pth', map_location='cpu')

# Create a new model with the correct architecture
new_model = DeepInfantModel()

# Try to map old layer names to new layer names
state_dict_mapping = {
    'c.0.weight': 'conv_layers.0.weight',
    'c.0.bias': 'conv_layers.0.bias',
    'c.1.weight': 'conv_layers.1.weight',
    'c.1.bias': 'conv_layers.1.bias',
    'c.1.running_mean': 'conv_layers.1.running_mean',
    'c.1.running_var': 'conv_layers.1.running_var',
    'c.1.num_batches_tracked': 'conv_layers.1.num_batches_tracked',
    'c.3.weight': 'conv_layers.3.weight',
    'c.3.bias': 'conv_layers.3.bias',
    'c.4.weight': 'conv_layers.4.weight',
    'c.4.bias': 'conv_layers.4.bias',
    'c.4.running_mean': 'conv_layers.4.running_mean',
    'c.4.running_var': 'conv_layers.4.running_var',
    'c.4.num_batches_tracked': 'conv_layers.4.num_batches_tracked',
    'c.6.weight': 'conv_layers.6.weight',
    'c.6.bias': 'conv_layers.6.bias',
    'c.7.weight': 'conv_layers.7.weight',
    'c.7.bias': 'conv_layers.7.bias',
    'c.7.running_mean': 'conv_layers.7.running_mean',
    'c.7.running_var': 'conv_layers.7.running_var',
    'c.7.num_batches_tracked': 'conv_layers.7.num_batches_tracked',
    'f.0.weight': 'classifier.0.weight',
    'f.0.bias': 'classifier.0.bias',
    'f.2.weight': 'classifier.3.weight',
    'f.2.bias': 'classifier.3.bias',
}

# Create new state dict with mapped keys
new_state_dict = {}
for old_key, value in old_checkpoint.items():
    if old_key in state_dict_mapping:
        new_key = state_dict_mapping[old_key]
        new_state_dict[new_key] = value
    elif old_key.startswith('l.'):
        # Map LSTM layers
        new_key = old_key.replace('l.', 'lstm.')
        new_state_dict[new_key] = value

# Load the mapped state dict
try:
    new_model.load_state_dict(new_state_dict, strict=False)
    print("Model weights loaded with some missing keys (expected for proj layer)")
except Exception as e:
    print(f"Error loading model: {e}")

# Save the converted model
torch.save(new_model.state_dict(), 'deepinfant_converted.pth')
print("Model converted and saved as deepinfant_converted.pth")
