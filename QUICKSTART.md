# DimViz Quick Start Guide

Get up and running with DimViz in 5 minutes!

## Installation

```bash
pip install dimviz
```

For colored output (recommended):
```bash
pip install dimviz[rich]
```

## Basic Usage

### 1. Track Your Model

```python
import torch
import torch.nn as nn
from dimviz import DimViz

# Your model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# Track shapes
with DimViz():
    x = torch.randn(4, 10)
    output = model(x)
```

**Output:**
```
╭──────┬───────────┬─────────────┬──────────────╮
│ Step │ Operation │ Input Shape │ Output Shape │
├──────┼───────────┼─────────────┼──────────────┤
│    1 │ linear    │ (4, 10)     │ (4, 20)      │
│    2 │ relu      │ (4, 20)     │ (4, 20)      │
│    3 │ linear    │ (4, 20)     │ (4, 5)       │
╰──────┴───────────┴─────────────┴──────────────╯
```

### 2. Use as a Decorator

```python
from dimviz import visualize

@visualize()
def forward_pass(model, x):
    return model(x)

output = forward_pass(model, x)
```

### 3. Track Only Shape Changes

```python
# Only log operations that change tensor shapes
with DimViz(verbose=False):
    output = model(x)
```

### 4. Track Memory Usage

```python
# See memory allocation per operation
with DimViz(track_memory=True):
    output = model(x)
```

### 5. Filter Operations

```python
# Only track specific operations
with DimViz(filter_ops=['linear', 'conv2d']):
    output = model(x)
```

### 6. Export Logs

```python
# Save logs for later analysis
with DimViz() as viz:
    output = model(x)

from dimviz import export_log
export_log(viz.get_log(), 'shapes.json')
export_log(viz.get_log(), 'shapes.csv')
```

## Common Use Cases

### Debug Shape Mismatches

```python
with DimViz(verbose=False):  # Only see shape changes
    try:
        output = problematic_model(x)
    except RuntimeError as e:
        print("Check the log above to see where shapes went wrong!")
```

### Understand Model Architecture

```python
# See how data flows through your model
with DimViz():
    output = complex_model(x)
```

### Optimize Memory

```python
# Find memory-hungry operations
with DimViz(track_memory=True, filter_ops=['conv2d', 'linear']):
    output = large_model(x)
```

## Tips & Tricks

### Tip 1: Reduce Noise
Use `verbose=False` to only see shape transformations:
```python
with DimViz(verbose=False):
    output = model(x)
```

### Tip 2: Focus on Important Ops
Filter to only track operations you care about:
```python
with DimViz(filter_ops=['matmul', 'conv2d', 'linear']):
    output = model(x)
```

### Tip 3: Limit Output
Limit the number of logged operations:
```python
with DimViz(max_entries=20):
    output = model(x)
```

### Tip 4: Hide Summary
Turn off the summary if you just want the table:
```python
with DimViz(show_summary=False):
    output = model(x)
```

### Tip 5: Compare Models
Compare shape flows between model versions:
```python
from dimviz.exporter import compare_logs

with DimViz() as viz1:
    out1 = model_v1(x)

with DimViz() as viz2:
    out2 = model_v2(x)

diff = compare_logs(viz1.get_log(), viz2.get_log())
```

## Real-World Example: CNN

```python
import torch
import torch.nn as nn
from dimviz import DimViz

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleCNN()
x = torch.randn(4, 3, 32, 32)

with DimViz(verbose=False):  # Only shape changes
    output = model(x)
```

**Output:**
```
╭──────┬────────────┬───────────────┬──────────────╮
│ Step │ Operation  │ Input Shape   │ Output Shape │
├──────┼────────────┼───────────────┼──────────────┤
│    1 │ conv2d     │ (4, 3, 32, 32)│ (4, 32, 32, 32) │
│    2 │ max_pool2d │ (4, 32, 32, 32)│ (4, 32, 16, 16) │
│    3 │ conv2d     │ (4, 32, 16, 16)│ (4, 64, 16, 16) │
│    4 │ max_pool2d │ (4, 64, 16, 16)│ (4, 64, 8, 8)   │
│    5 │ view       │ (4, 64, 8, 8) │ (4, 4096)    │
│    6 │ linear     │ (4, 4096)     │ (4, 10)      │
╰──────┴────────────┴───────────────┴──────────────╯
```

Perfect! You can now see exactly how the spatial dimensions reduce through pooling and how the tensor is flattened before the fully connected layer.

## Next Steps

- Check out the [full documentation](README.md)
- See more [examples](examples/)
- Read about [contributing](CONTRIBUTING.md)
- Report [issues](https://github.com/yourusername/dimviz/issues)

## Need Help?

- 📖 Read the [README](README.md)
- 🐛 Found a bug? [Open an issue](https://github.com/yourusername/dimviz/issues)
- 💡 Have a feature idea? [Start a discussion](https://github.com/yourusername/dimviz/discussions)

Happy debugging! 🔍✨
