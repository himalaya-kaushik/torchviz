"""
DimViz - Tensor Shape Visualization for PyTorch
A lightweight debugging tool for tracking tensor shape transformations in PyTorch models.
"""

__version__ = "0.1.0"
__author__ = "Himalaya"
__license__ = "MIT"

from .tracker import DimViz, visualize, DimVizTracker
from .exporter import export_log

__all__ = ["DimViz", "visualize", "DimVizTracker", "export_log"]
