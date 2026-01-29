"""Core tracking functionality for DimViz."""

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from tabulate import tabulate
import functools
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict


FRIENDLY_NAMES = {
    't': 'transpose',
    'mm': 'matmul',
    'addmm': 'linear_proj',
    'bmm': 'batch_matmul',
    'select': 'slice_index',
    'unsafe_split': 'split',
    'unsafe_view': 'view',
    'unbind': 'unstack',
    'cat': 'concat',
    'mul': 'multiply',
    'div': 'divide',
    'sub': 'subtract',
    'add': 'add',
    'sigmoid': 'sigmoid',
    'tanh': 'tanh',
    'relu': 'relu',
    'gelu': 'gelu',
    'softmax': 'softmax',
    'layer_norm': 'layer_norm',
    'batch_norm': 'batch_norm',
    'conv2d': 'conv2d',
    'max_pool2d': 'max_pool2d',
    'avg_pool2d': 'avg_pool2d',
    'dropout': 'dropout',
    'embedding': 'embedding',
    'linear': 'linear',
    'reshape': 'reshape',
    'flatten': 'flatten',
    'squeeze': 'squeeze',
    'unsqueeze': 'unsqueeze',
    'permute': 'permute',
    'contiguous': 'contiguous',
    'clone': 'clone',
    'expand': 'expand',
    'repeat': 'repeat',
    'mean': 'mean',
    'sum': 'sum',
    'std': 'std',
    'var': 'variance',
}

IGNORED_OPS = {
    'aten::size', 'aten::stride', 'aten::storage_offset', 'aten::is_floating_point',
    'aten::is_complex', 'aten::is_conj', 'aten::numel', 'aten::dim',
    'aten::detach', 'aten::empty', 'aten::empty_like', 'aten::as_strided',
    'aten::_local_scalar_dense', 'aten::_to_copy', 'aten::item',
    'aten::is_same_size', 'aten::is_nonzero', 'aten::scalar_tensor',
}


class DimVizTracker(TorchDispatchMode):
    """
    Core tracker that intercepts PyTorch operations to log shape transformations.
    
    Args:
        verbose: If True, log all operations. If False, only log shape changes.
        track_memory: If True, track memory allocation for each operation.
        filter_ops: Optional list of operation names to track (if None, track all).
        max_entries: Maximum number of log entries (None = unlimited).
    """
    
    def __init__(
        self,
        verbose: bool = True,
        track_memory: bool = False,
        filter_ops: Optional[List[str]] = None,
        max_entries: Optional[int] = None
    ):
        self.log = []
        self.step = 0
        self.verbose = verbose
        self.track_memory = track_memory
        self.filter_ops = set(filter_ops) if filter_ops else None
        self.max_entries = max_entries
        self.op_counts = defaultdict(int)
        self.start_time = None
        self.memory_stats = []
        
    def _format_shape(self, obj: Any) -> str:
        """Format tensor shapes into readable strings."""
        if isinstance(obj, torch.Tensor):
            shape = tuple(obj.shape)
            if len(shape) == 0:
                return "scalar"
            return str(shape)
        if isinstance(obj, (list, tuple)):
            shapes = [
                self._format_shape(x) 
                for x in obj 
                if isinstance(x, (torch.Tensor, list, tuple))
            ]
            if shapes:
                return " | ".join(shapes)
        return ""
    
    def _get_all_input_shapes(self, args: Tuple, kwargs: Dict) -> str:
        """Extract shapes from all tensor arguments."""
        shapes = []
        
        # Get shapes from args
        for arg in args:
            if isinstance(arg, torch.Tensor):
                shapes.append(self._format_shape(arg))
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, torch.Tensor):
                        shapes.append(self._format_shape(item))
        
        # Get shapes from kwargs
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                shapes.append(self._format_shape(value))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_shapes = []
        for shape in shapes:
            if shape and shape not in seen:
                seen.add(shape)
                unique_shapes.append(shape)
        
        return " + ".join(unique_shapes) if unique_shapes else ""
    
    def _get_memory_mb(self, obj: Any) -> float:
        """Calculate memory usage in MB for tensors."""
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement() / (1024 ** 2)
        elif isinstance(obj, (list, tuple)):
            return sum(self._get_memory_mb(x) for x in obj if isinstance(x, torch.Tensor))
        return 0.0
    
    def _should_log(self, op_name: str, in_shape: str, out_shape: str) -> bool:
        """Determine if operation should be logged."""
        # Check max entries limit
        if self.max_entries and len(self.log) >= self.max_entries:
            return False
        
        # Check filter
        if self.filter_ops and op_name not in self.filter_ops:
            return False
        
        # Check verbose mode
        if not in_shape or not out_shape:
            return False
            
        shape_changed = (in_shape != out_shape)
        return self.verbose or shape_changed
    
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Intercept torch operations to log shape transformations."""
        if kwargs is None:
            kwargs = {}
        
        op_name = func._schema.name
        
        # Execute the operation
        output = func(*args, **kwargs)
        
        # Skip ignored operations
        if op_name in IGNORED_OPS:
            return output
        
        # Process operation name
        raw_name = op_name.replace("aten::", "")
        lookup_name = raw_name.rstrip('_')
        friendly_base = FRIENDLY_NAMES.get(lookup_name, raw_name)
        
        # Preserve in-place indicator
        if raw_name.endswith('_') and not friendly_base.endswith('_'):
            clean_name = friendly_base + "_"
        else:
            clean_name = friendly_base
        
        # Get shapes
        in_shape = self._get_all_input_shapes(args, kwargs)
        out_shape = self._format_shape(output)
        
        # Track operation counts
        self.op_counts[clean_name] += 1
        
        # Log if conditions met
        if self._should_log(clean_name, in_shape, out_shape):
            self.step += 1
            
            log_entry = [self.step, clean_name, in_shape, out_shape]
            
            # Add memory info if tracking
            if self.track_memory:
                mem_in = self._get_memory_mb(args[0]) if args else 0.0
                mem_out = self._get_memory_mb(output)
                log_entry.extend([f"{mem_in:.2f}MB", f"{mem_out:.2f}MB"])
                self.memory_stats.append({
                    'step': self.step,
                    'op': clean_name,
                    'mem_in': mem_in,
                    'mem_out': mem_out,
                    'mem_delta': mem_out - mem_in
                })
            
            self.log.append(log_entry)
        
        return output
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of tracked operations."""
        total_ops = sum(self.op_counts.values())
        summary = {
            'total_operations': total_ops,
            'unique_operations': len(self.op_counts),
            'logged_operations': len(self.log),
            'operation_counts': dict(self.op_counts),
        }
        
        if self.track_memory and self.memory_stats:
            total_mem_delta = sum(s['mem_delta'] for s in self.memory_stats)
            max_mem = max(s['mem_out'] for s in self.memory_stats)
            summary.update({
                'total_memory_delta': f"{total_mem_delta:.2f}MB",
                'peak_memory': f"{max_mem:.2f}MB",
            })
        
        return summary


class DimViz:
    """
    Context manager for tracking tensor shape transformations.
    
    Usage:
        Basic tracking:
            with DimViz():
                model(x)
        
        Only shape changes:
            with DimViz(verbose=False):
                model(x)
        
        With memory tracking:
            with DimViz(track_memory=True):
                model(x)
        
        Filter specific operations:
            with DimViz(filter_ops=['matmul', 'conv2d']):
                model(x)
    
    Args:
        verbose: If True, log all operations. If False, only log shape changes.
        track_memory: If True, track memory allocation for each operation.
        filter_ops: Optional list of operation names to track.
        max_entries: Maximum number of log entries (None = unlimited).
        show_summary: If True, print summary statistics at the end.
    """
    
    def __init__(
        self,
        verbose: bool = True,
        track_memory: bool = False,
        filter_ops: Optional[List[str]] = None,
        max_entries: Optional[int] = None,
        show_summary: bool = True
    ):
        self.tracker = DimVizTracker(
            verbose=verbose,
            track_memory=track_memory,
            filter_ops=filter_ops,
            max_entries=max_entries
        )
        self.show_summary = show_summary
        self._start_time = None
    
    def __enter__(self):
        """Start tracking operations."""
        self._start_time = time.time()
        print(f"\n[DimViz] 🟢 Tracking Started...")
        self.tracker.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Stop tracking and display results."""
        self.tracker.__exit__(exc_type, exc_value, traceback)
        elapsed = time.time() - self._start_time
        
        print(f"[DimViz] 🔴 Tracking Finished. (Elapsed: {elapsed:.2f}s)")
        
        if exc_type:
            print(f"\n[DimViz] ⚠️  CRASH DETECTED: {exc_value}")
            return False
        
        # Display the table
        self._display_table()
        
        # Display summary if requested
        if self.show_summary:
            self._display_summary()
        
        return False
    
    def _display_table(self):
        """Display the operations table."""
        if not self.tracker.log:
            print("\n[DimViz] ℹ️  No operations logged.")
            return
        
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box
            
            console = Console()
            table = Table(title="🔍 Dimension Flow Log", box=box.ROUNDED)
            
            table.add_column("Step", style="dim", justify="right")
            table.add_column("Operation", style="cyan")
            table.add_column("Input Shape(s)", style="yellow")
            table.add_column("Output Shape", style="bold green")
            
            if self.tracker.track_memory:
                table.add_column("Mem In", style="magenta")
                table.add_column("Mem Out", style="magenta")
            
            for row in self.tracker.log:
                table.add_row(*[str(cell) for cell in row])
            
            console.print(table)
            
        except ImportError:
            # Fallback to tabulate
            headers = ["Step", "Op", "Input", "Output"]
            if self.tracker.track_memory:
                headers.extend(["Mem In", "Mem Out"])
            
            print("\n" + tabulate(
                self.tracker.log,
                headers=headers,
                tablefmt="fancy_grid"
            ))
    
    def _display_summary(self):
        """Display summary statistics."""
        summary = self.tracker.get_summary()
        
        print(f"\n[DimViz] 📊 Summary:")
        print(f"  • Total Operations: {summary['total_operations']}")
        print(f"  • Unique Operations: {summary['unique_operations']}")
        print(f"  • Logged Operations: {summary['logged_operations']}")
        
        if self.tracker.track_memory and 'peak_memory' in summary:
            print(f"  • Peak Memory: {summary['peak_memory']}")
            print(f"  • Total Memory Delta: {summary['total_memory_delta']}")
        
        # Show top operations
        top_ops = sorted(
            summary['operation_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        if top_ops:
            print(f"\n  Top Operations:")
            for op, count in top_ops:
                print(f"    - {op}: {count}x")
    
    def get_log(self) -> List[List[Any]]:
        """Get the raw log data."""
        return self.tracker.log
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return self.tracker.get_summary()


def visualize(
    verbose: bool = True,
    track_memory: bool = False,
    filter_ops: Optional[List[str]] = None,
    show_summary: bool = True
):
    """
    Decorator for tracking tensor shape transformations in a function.
    
    Usage:
        @visualize()
        def forward(x):
            return model(x)
        
        @visualize(verbose=False, track_memory=True)
        def train_step(batch):
            return loss.backward()
    
    Args:
        verbose: If True, log all operations. If False, only log shape changes.
        track_memory: If True, track memory allocation.
        filter_ops: Optional list of operation names to track.
        show_summary: If True, print summary statistics.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with DimViz(
                verbose=verbose,
                track_memory=track_memory,
                filter_ops=filter_ops,
                show_summary=show_summary
            ):
                return func(*args, **kwargs)
        return wrapper
    return decorator
