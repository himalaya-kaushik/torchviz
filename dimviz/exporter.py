"""Export functionality for DimViz logs."""

import json
import csv
from pathlib import Path
from typing import List, Any, Dict, Optional
from datetime import datetime


def export_log(
    log_data: List[List[Any]],
    filepath: str,
    format: str = "auto",
    include_metadata: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Export DimViz log to various formats.
    
    Args:
        log_data: The log data from DimViz tracker.
        filepath: Output file path.
        format: Export format ('json', 'csv', 'txt', or 'auto' to infer from extension).
        include_metadata: Whether to include metadata (timestamp, etc.).
        metadata: Additional metadata to include in the export.
    
    Raises:
        ValueError: If format is unsupported or cannot be inferred.
    
    Example:
        with DimViz() as viz:
            model(x)
        export_log(viz.get_log(), 'output.json')
    """
    filepath = Path(filepath)
    
    # Infer format from extension
    if format == "auto":
        ext = filepath.suffix.lower()
        format_map = {
            '.json': 'json',
            '.csv': 'csv',
            '.txt': 'txt',
            '.log': 'txt',
        }
        format = format_map.get(ext)
        if format is None:
            raise ValueError(
                f"Cannot infer format from extension '{ext}'. "
                f"Supported: {list(format_map.keys())}"
            )
    
    # Prepare metadata
    if include_metadata:
        if metadata is None:
            metadata = {}
        metadata.setdefault('timestamp', datetime.now().isoformat())
        metadata.setdefault('total_operations', len(log_data))
    
    # Export based on format
    if format == 'json':
        _export_json(log_data, filepath, metadata if include_metadata else None)
    elif format == 'csv':
        _export_csv(log_data, filepath)
    elif format == 'txt':
        _export_txt(log_data, filepath, metadata if include_metadata else None)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"[DimViz] 💾 Log exported to: {filepath}")


def _export_json(log_data: List[List[Any]], filepath: Path, metadata: Optional[Dict]) -> None:
    """Export log as JSON."""
    # Convert log to structured format
    headers = ["step", "operation", "input_shape", "output_shape"]
    
    # Check if memory data is present
    if log_data and len(log_data[0]) > 4:
        headers.extend(["memory_in", "memory_out"])
    
    structured_log = [
        dict(zip(headers, row))
        for row in log_data
    ]
    
    output = {
        "log": structured_log
    }
    
    if metadata:
        output["metadata"] = metadata
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)


def _export_csv(log_data: List[List[Any]], filepath: Path) -> None:
    """Export log as CSV."""
    if not log_data:
        return
    
    headers = ["step", "operation", "input_shape", "output_shape"]
    
    # Check if memory data is present
    if len(log_data[0]) > 4:
        headers.extend(["memory_in", "memory_out"])
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(log_data)


def _export_txt(log_data: List[List[Any]], filepath: Path, metadata: Optional[Dict]) -> None:
    """Export log as formatted text."""
    with open(filepath, 'w') as f:
        if metadata:
            f.write("DimViz Log Export\n")
            f.write("=" * 80 + "\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
            f.write("=" * 80 + "\n\n")
        
        # Write header
        headers = ["Step", "Operation", "Input Shape", "Output Shape"]
        if log_data and len(log_data[0]) > 4:
            headers.extend(["Mem In", "Mem Out"])
        
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in log_data:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Write header
        header_line = " | ".join(
            h.ljust(w) for h, w in zip(headers, col_widths)
        )
        f.write(header_line + "\n")
        f.write("-" * len(header_line) + "\n")
        
        # Write rows
        for row in log_data:
            line = " | ".join(
                str(cell).ljust(w) for cell, w in zip(row, col_widths)
            )
            f.write(line + "\n")


def compare_logs(
    log1: List[List[Any]],
    log2: List[List[Any]],
    name1: str = "Run 1",
    name2: str = "Run 2"
) -> Dict[str, Any]:
    """
    Compare two DimViz logs to find differences.
    
    Args:
        log1: First log data.
        log2: Second log data.
        name1: Name for first run.
        name2: Name for second run.
    
    Returns:
        Dictionary containing comparison results.
    
    Example:
        with DimViz() as viz1:
            model_v1(x)
        log1 = viz1.get_log()
        
        with DimViz() as viz2:
            model_v2(x)
        log2 = viz2.get_log()
        
        diff = compare_logs(log1, log2, "v1", "v2")
    """
    comparison = {
        'name1': name1,
        'name2': name2,
        'length_diff': len(log2) - len(log1),
        'differences': []
    }
    
    # Compare step by step
    min_len = min(len(log1), len(log2))
    for i in range(min_len):
        row1, row2 = log1[i], log2[i]
        
        # Compare operation names
        if row1[1] != row2[1]:
            comparison['differences'].append({
                'step': i + 1,
                'type': 'operation',
                name1: row1[1],
                name2: row2[1]
            })
        
        # Compare output shapes
        elif row1[3] != row2[3]:
            comparison['differences'].append({
                'step': i + 1,
                'type': 'shape',
                'operation': row1[1],
                f'{name1}_shape': row1[3],
                f'{name2}_shape': row2[3]
            })
    
    # Note if one log is longer
    if len(log1) != len(log2):
        comparison['extra_ops'] = {
            'in': name1 if len(log1) > len(log2) else name2,
            'count': abs(len(log1) - len(log2))
        }
    
    return comparison
