"""
Example usage of DimViz for debugging PyTorch models.
"""

import torch
import torch.nn as nn
from dimviz import DimViz, visualize, export_log


# Example 1: Basic CNN Model
print("=" * 80)
print("Example 1: Basic CNN Model")
print("=" * 80)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN()
x = torch.randn(4, 3, 32, 32)

with DimViz(verbose=True):
    output = model(x)

print(f"\nFinal output shape: {output.shape}")


# Example 2: Only Shape Changes
print("\n" + "=" * 80)
print("Example 2: Track Only Shape Changes")
print("=" * 80)

with DimViz(verbose=False):
    output = model(x)


# Example 3: Memory Tracking
print("\n" + "=" * 80)
print("Example 3: Memory Tracking")
print("=" * 80)

with DimViz(track_memory=True, verbose=False):
    output = model(x)


# Example 4: Filter Specific Operations
print("\n" + "=" * 80)
print("Example 4: Filter Specific Operations (conv2d and linear only)")
print("=" * 80)

with DimViz(filter_ops=['conv2d', 'linear']):
    output = model(x)


# Example 5: Using Decorator
print("\n" + "=" * 80)
print("Example 5: Using @visualize Decorator")
print("=" * 80)

@visualize(verbose=False)
def inference(model, x):
    """Run inference with shape tracking."""
    return model(x)

result = inference(model, x)


# Example 6: Export Logs
print("\n" + "=" * 80)
print("Example 6: Export Logs to Files")
print("=" * 80)

with DimViz(show_summary=False) as viz:
    output = model(x)

# Export to different formats
export_log(viz.get_log(), 'cnn_shapes.json')
export_log(viz.get_log(), 'cnn_shapes.csv')
export_log(viz.get_log(), 'cnn_shapes.txt')

print("\n✅ Logs exported to:")
print("  - cnn_shapes.json")
print("  - cnn_shapes.csv")
print("  - cnn_shapes.txt")


# Example 7: Transformer-like Model
print("\n" + "=" * 80)
print("Example 7: Transformer-like Attention Mechanism")
print("=" * 80)

class SimpleAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (x.size(-1) ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn_weights, V)
        out = self.out(out)
        return out


attention_model = SimpleAttention(embed_dim=64)
x = torch.randn(2, 10, 64)  # (batch=2, seq_len=10, embed_dim=64)

with DimViz(verbose=False):
    output = attention_model(x)

print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")


# Example 8: Compare Two Models
print("\n" + "=" * 80)
print("Example 8: Compare Two Model Versions")
print("=" * 80)

from dimviz.exporter import compare_logs

# Model v1 - Simple
class ModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Model v2 - With extra layer
class ModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 15)  # Extra layer
        self.fc3 = nn.Linear(15, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


x = torch.randn(4, 10)

# Track both models
with DimViz(show_summary=False) as viz1:
    output1 = ModelV1()(x)

with DimViz(show_summary=False) as viz2:
    output2 = ModelV2()(x)

# Compare
comparison = compare_logs(viz1.get_log(), viz2.get_log(), "ModelV1", "ModelV2")

print(f"\n📊 Comparison Results:")
print(f"  Length difference: {comparison['length_diff']} operations")
if 'extra_ops' in comparison:
    print(f"  Extra operations in {comparison['extra_ops']['in']}: {comparison['extra_ops']['count']}")
if comparison['differences']:
    print(f"  Found {len(comparison['differences'])} differences")


# Example 9: Debugging Shape Mismatch
print("\n" + "=" * 80)
print("Example 9: Debugging a Shape Mismatch (Intentional Error)")
print("=" * 80)

class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16 * 30 * 30, 10)  # Wrong size!
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # This will crash!
        return x


buggy_model = BuggyModel()
x = torch.randn(2, 3, 32, 32)

try:
    with DimViz(verbose=False):
        output = buggy_model(x)
except RuntimeError as e:
    print(f"\n⚠️  Caught shape mismatch!")
    print(f"Error: {str(e)[:100]}...")
    print("\n💡 DimViz helps you see where the mismatch occurred!")


print("\n" + "=" * 80)
print("All examples completed! 🎉")
print("=" * 80)
