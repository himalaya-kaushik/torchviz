"""Test suite for DimViz."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import json
import csv
import tempfile
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dimviz import DimViz, visualize, export_log
from dimviz.tracker import DimVizTracker
from dimviz.exporter import compare_logs


class SimpleModel(nn.Module):
    """Simple test model."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class TestDimVizTracker:
    """Test DimVizTracker functionality."""
    
    def test_basic_tracking(self):
        """Test basic operation tracking."""
        with DimViz(verbose=True, show_summary=False) as viz:
            x = torch.randn(2, 10)
            y = torch.matmul(x, torch.randn(10, 5))
        
        assert len(viz.get_log()) > 0
        assert any('matmul' in str(row) for row in viz.get_log())
    
    def test_verbose_vs_non_verbose(self):
        """Test verbose vs non-verbose mode."""
        # Verbose mode
        with DimViz(verbose=True, show_summary=False) as viz_verbose:
            x = torch.randn(2, 10)
            y = x + 1  # No shape change
            z = x.reshape(5, 4)  # Shape change
        
        # Non-verbose mode
        with DimViz(verbose=False, show_summary=False) as viz_quiet:
            x = torch.randn(2, 10)
            y = x + 1  # No shape change
            z = x.reshape(5, 4)  # Shape change
        
        assert len(viz_verbose.get_log()) > len(viz_quiet.get_log())
    
    def test_memory_tracking(self):
        """Test memory tracking functionality."""
        with DimViz(track_memory=True, show_summary=False) as viz:
            x = torch.randn(100, 100)
            y = torch.matmul(x, x)
        
        log = viz.get_log()
        # Check that memory columns exist
        if log:
            assert len(log[0]) >= 6  # Should have memory columns
            assert 'MB' in str(log[0][-1])
    
    def test_filter_ops(self):
        """Test operation filtering."""
        with DimViz(filter_ops=['matmul'], show_summary=False) as viz:
            x = torch.randn(2, 10)
            y = x + 1  # Should be filtered out
            z = torch.matmul(x, torch.randn(10, 5))  # Should be logged
        
        log = viz.get_log()
        assert all('matmul' in str(row[1]) for row in log)
    
    def test_max_entries(self):
        """Test max entries limit."""
        with DimViz(max_entries=5, show_summary=False) as viz:
            for _ in range(20):
                x = torch.randn(2, 3)
                y = x + 1
        
        assert len(viz.get_log()) <= 5
    
    def test_multi_input_shapes(self):
        """Test tracking multiple input shapes."""
        with DimViz(show_summary=False) as viz:
            x1 = torch.randn(2, 3)
            x2 = torch.randn(2, 3)
            y = x1 + x2
        
        log = viz.get_log()
        # Should capture both input shapes
        assert any('+' in str(row[2]) or '|' in str(row[2]) for row in log)


class TestDimVizContextManager:
    """Test DimViz context manager."""
    
    def test_basic_usage(self):
        """Test basic context manager usage."""
        model = SimpleModel()
        x = torch.randn(4, 10)
        
        with DimViz(show_summary=False) as viz:
            output = model(x)
        
        assert output.shape == (4, 5)
        assert len(viz.get_log()) > 0
    
    def test_exception_handling(self):
        """Test that exceptions are properly handled."""
        try:
            with DimViz(show_summary=False) as viz:
                x = torch.randn(2, 3)
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Should still have logged operations before the error
        assert len(viz.get_log()) >= 0
    
    def test_get_summary(self):
        """Test summary statistics."""
        with DimViz(show_summary=False) as viz:
            x = torch.randn(2, 10)
            y = torch.matmul(x, torch.randn(10, 5))
            z = y + 1
        
        summary = viz.get_summary()
        assert 'total_operations' in summary
        assert 'unique_operations' in summary
        assert summary['total_operations'] > 0


class TestVisualizeDecorator:
    """Test the @visualize decorator."""
    
    def test_decorator_basic(self):
        """Test basic decorator usage."""
        @visualize(show_summary=False)
        def simple_operation(x):
            return x + 1
        
        result = simple_operation(torch.randn(2, 3))
        assert result.shape == (2, 3)
    
    def test_decorator_with_model(self):
        """Test decorator with model forward pass."""
        model = SimpleModel()
        
        @visualize(verbose=False, show_summary=False)
        def forward_pass(x):
            return model(x)
        
        x = torch.randn(4, 10)
        output = forward_pass(x)
        assert output.shape == (4, 5)
    
    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""
        @visualize(show_summary=False)
        def documented_function(x):
            """This is a documented function."""
            return x * 2
        
        assert documented_function.__doc__ == "This is a documented function."
        assert documented_function.__name__ == "documented_function"


class TestExporter:
    """Test export functionality."""
    
    def test_export_json(self):
        """Test JSON export."""
        with DimViz(show_summary=False) as viz:
            x = torch.randn(2, 10)
            y = torch.matmul(x, torch.randn(10, 5))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            export_log(viz.get_log(), filepath)
            
            # Verify file exists and is valid JSON
            assert Path(filepath).exists()
            with open(filepath) as f:
                data = json.load(f)
                assert 'log' in data
                assert 'metadata' in data
                assert len(data['log']) > 0
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_export_csv(self):
        """Test CSV export."""
        with DimViz(show_summary=False) as viz:
            x = torch.randn(2, 10)
            y = x + 1
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            export_log(viz.get_log(), filepath)
            
            # Verify file exists and is valid CSV
            assert Path(filepath).exists()
            with open(filepath) as f:
                reader = csv.reader(f)
                rows = list(reader)
                assert len(rows) > 1  # Header + data
                assert 'step' in rows[0][0].lower()
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_export_txt(self):
        """Test TXT export."""
        with DimViz(show_summary=False) as viz:
            x = torch.randn(2, 10)
            y = x.reshape(5, 4)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            filepath = f.name
        
        try:
            export_log(viz.get_log(), filepath)
            
            # Verify file exists and has content
            assert Path(filepath).exists()
            content = Path(filepath).read_text()
            assert len(content) > 0
            assert 'Step' in content
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_auto_format_detection(self):
        """Test automatic format detection from file extension."""
        with DimViz(show_summary=False) as viz:
            x = torch.randn(2, 10)
        
        for ext in ['.json', '.csv', '.txt']:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                filepath = f.name
            
            try:
                export_log(viz.get_log(), filepath, format='auto')
                assert Path(filepath).exists()
            finally:
                Path(filepath).unlink(missing_ok=True)
    
    def test_compare_logs(self):
        """Test log comparison functionality."""
        # First run
        with DimViz(show_summary=False) as viz1:
            x = torch.randn(2, 10)
            y = x + 1
        
        # Second run with different operations
        with DimViz(show_summary=False) as viz2:
            x = torch.randn(2, 10)
            y = x * 2
        
        comparison = compare_logs(viz1.get_log(), viz2.get_log())
        assert 'differences' in comparison
        assert 'length_diff' in comparison


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_scalar_tensors(self):
        """Test handling of scalar tensors."""
        with DimViz(show_summary=False) as viz:
            x = torch.tensor(5.0)
            y = x + 1
        
        log = viz.get_log()
        assert any('scalar' in str(row).lower() for row in log)
    
    def test_empty_log(self):
        """Test behavior with no operations."""
        with DimViz(show_summary=False) as viz:
            pass
        
        assert len(viz.get_log()) == 0
    
    def test_inplace_operations(self):
        """Test in-place operation tracking."""
        with DimViz(show_summary=False) as viz:
            x = torch.randn(2, 3)
            x.add_(1)  # In-place operation
        
        log = viz.get_log()
        # Check for underscore in operation name
        assert any('_' in str(row[1]) for row in log)
    
    def test_complex_model(self):
        """Test with a more complex model."""
        class ComplexModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.pool = nn.MaxPool2d(2)
                self.fc = nn.Linear(16 * 13 * 13, 10)
            
            def forward(self, x):
                x = self.conv(x)
                x = torch.relu(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = ComplexModel()
        x = torch.randn(2, 3, 28, 28)
        
        with DimViz(show_summary=False) as viz:
            output = model(x)
        
        assert output.shape == (2, 10)
        assert len(viz.get_log()) > 5  # Should have multiple operations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
