"""
Memory optimization utilities for handling large datasets efficiently.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm


@dataclass
class ChunkingStrategy:
    chunk_size: int
    num_chunks: int
    estimated_memory_per_chunk: int
    total_rows: int

    def __str__(self):
        return (
            f"ChunkingStrategy(chunk_size={self.chunk_size:,} rows, "
            f"num_chunks={self.num_chunks}, "
            f"estimated_memory_per_chunk={self.estimated_memory_per_chunk / 1024**2:.2f}MB, "
            f"total_rows={self.total_rows:,})"
        )


def get_system_memory() -> int:
    """Get available system memory in bytes."""
    return psutil.virtual_memory().available


def estimate_row_size(df: pd.DataFrame) -> float:
    """Estimate memory size per row in bytes."""
    return df.memory_usage(deep=True).sum() / len(df)


def calculate_optimal_chunks(
    file_path: str,
    available_memory: Optional[int] = None,
    safety_factor: float = 0.7,
    operation_type: str = "read",
    sample_size: int = 10000,
) -> ChunkingStrategy:
    """
    Calculate optimal chunk size for processing large files.

    Args:
        file_path: Path to the file to process
        available_memory: Available memory in bytes (if None, will auto-detect)
        safety_factor: Fraction of available memory to use
        operation_type: Type of operation ('read', 'transform', 'groupby')
        sample_size: Number of rows to sample for size estimation

    Returns:
        ChunkingStrategy object with chunking details
    """
    # Get available memory if not provided
    if available_memory is None:
        available_memory = get_system_memory()

    # Adjust available memory by safety factor and operation type
    operation_memory_factors = {
        "read": 1.0,
        "transform": 2.0,  # Need space for input and output
        "groupby": 3.0,  # GroupBy can create large temporary objects
    }

    adjusted_memory = (available_memory * safety_factor) / operation_memory_factors.get(
        operation_type, 1.0
    )

    # Read a sample to estimate row size
    sample_df = pd.read_csv(file_path, nrows=sample_size)
    row_size = estimate_row_size(sample_df)

    # Count total rows in file
    total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header row

    # Calculate chunk size
    chunk_size = int(adjusted_memory / row_size)
    chunk_size = min(chunk_size, total_rows)  # Don't exceed total rows
    num_chunks = int(np.ceil(total_rows / chunk_size))

    return ChunkingStrategy(
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        estimated_memory_per_chunk=int(chunk_size * row_size),
        total_rows=total_rows,
    )


def optimize_dtypes(
    df: pd.DataFrame,
    aggressive: bool = False,
    categorical_threshold: float = 0.5,
    analyze_string_patterns: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Optimize DataFrame memory usage by choosing appropriate dtypes.

    Args:
        df: Input DataFrame
        aggressive: If True, might sacrifice some precision for memory savings
        categorical_threshold: Max ratio of unique values to total values for categorical conversion
        analyze_string_patterns: Look for date patterns and numbers in strings

    Returns:
        Tuple of (optimized DataFrame, optimization report)
    """
    original_memory = df.memory_usage(deep=True).sum()
    optimization_report = {"original_memory_mb": original_memory / 1024**2}

    # Copy DataFrame to avoid modifying original
    df_opt = df.copy()

    # Optimize numeric columns
    for col in df_opt.select_dtypes(include=["int", "float"]).columns:
        col_series = df_opt[col]

        # For integers
        if pd.api.types.is_integer_dtype(col_series):
            col_min, col_max = col_series.min(), col_series.max()

            # Find the smallest integer type that can hold the data
            for dtype in [np.int8, np.int16, np.int32, np.int64]:
                if col_min >= np.iinfo(dtype).min and col_max <= np.iinfo(dtype).max:
                    df_opt[col] = col_series.astype(dtype)
                    break

        # For floats
        elif aggressive and pd.api.types.is_float_dtype(col_series):
            df_opt[col] = col_series.astype(np.float32)

    # Optimize string columns
    for col in df_opt.select_dtypes(include=["object"]).columns:
        col_series = df_opt[col]

        # Check for categorical conversion
        unique_ratio = len(col_series.unique()) / len(col_series)
        if unique_ratio <= categorical_threshold:
            df_opt[col] = col_series.astype("category")
            continue

        if analyze_string_patterns:
            # Try converting to datetime
            try:
                pd.to_datetime(col_series, errors="raise")
                df_opt[col] = pd.to_datetime(col_series)
                continue
            except (ValueError, TypeError):
                pass

            # Try converting to numeric
            if aggressive:
                try:
                    numeric_series = pd.to_numeric(col_series, errors="raise")
                    df_opt[col] = numeric_series
                    continue
                except (ValueError, TypeError):
                    pass

    final_memory = df_opt.memory_usage(deep=True).sum()
    optimization_report.update(
        {
            "final_memory_mb": final_memory / 1024**2,
            "memory_savings_mb": (original_memory - final_memory) / 1024**2,
            "memory_savings_pct": (1 - final_memory / original_memory) * 100,
        }
    )

    return df_opt, optimization_report


class StreamingOperationManager:
    """
    Manage streaming operations on large datasets with progress tracking and memory management.
    """

    def __init__(self, memory_limit: Optional[int] = None):
        self.memory_limit = memory_limit or get_system_memory() * 0.8

    def process_large_file(
        self,
        input_path: str,
        operation: callable,
        chunk_handler: callable = None,
        show_progress: bool = True,
        checkpoint: bool = True,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Process a large file in chunks with progress tracking and memory management.

        Args:
            input_path: Path to input file
            operation: Function to apply to each chunk
            chunk_handler: Function to handle processed chunks (e.g., save to disk)
            show_progress: Whether to show progress bar
            checkpoint: Whether to save checkpoints
            checkpoint_path: Path to save checkpoints
        """
        # Calculate chunking strategy
        strategy = calculate_optimal_chunks(
            input_path, available_memory=self.memory_limit, safety_factor=0.7
        )

        # Initialize checkpoint
        last_processed_chunk = 0
        if checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                last_processed_chunk = int(f.read())

        chunks_iterator = pd.read_csv(input_path, chunksize=strategy.chunk_size)

        if show_progress:
            chunks_iterator = tqdm(chunks_iterator, total=strategy.num_chunks)

        for chunk_idx, chunk in enumerate(chunks_iterator):
            if chunk_idx < last_processed_chunk:
                continue

            try:
                # Process chunk
                processed_chunk = operation(chunk)

                # Handle processed chunk
                if chunk_handler:
                    chunk_handler(processed_chunk, chunk_idx)

                # Save checkpoint
                if checkpoint and checkpoint_path:
                    with open(checkpoint_path, "w") as f:
                        f.write(str(chunk_idx + 1))

                # Force garbage collection after each chunk
                del processed_chunk
                del chunk

            except Exception as e:
                print(f"Error processing chunk {chunk_idx}: {str(e)}")
                raise


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage of the Python process.

    Returns:
        Dict with memory usage in different units (bytes, MB, GB)
    """
    process = psutil.Process(os.getpid())
    bytes_usage = process.memory_info().rss

    return {
        "bytes": bytes_usage,
        "mb": bytes_usage / (1024 * 1024),
        "gb": bytes_usage / (1024 * 1024 * 1024),
    }


def get_dataframe_memory(df: pd.DataFrame) -> Dict[str, Union[float, Dict]]:
    """
    Get memory usage details for a pandas DataFrame.

    Args:
        df: pandas DataFrame to analyze

    Returns:
        Dict containing total memory usage and per-column memory usage
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory_bytes = memory_usage.sum()

    # Get per-column memory usage
    column_memory = {
        col: {
            "bytes": memory_usage[i],
            "mb": memory_usage[i] / (1024 * 1024),
            "gb": memory_usage[i] / (1024 * 1024 * 1024),
        }
        for i, col in enumerate(df.columns)
    }

    return {
        "total": {
            "bytes": total_memory_bytes,
            "mb": total_memory_bytes / (1024 * 1024),
            "gb": total_memory_bytes / (1024 * 1024 * 1024),
        },
        "columns": column_memory,
    }


def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize memory usage of a DataFrame by converting data types.

    Args:
        df: pandas DataFrame to optimize

    Returns:
        Optimized pandas DataFrame
    """
    df_optimized = df.copy()

    for col in df_optimized.columns:
        # Optimize integers
        if df_optimized[col].dtype in ["int64", "int32"]:
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()

            if c_min >= 0:
                if c_max < 255:
                    df_optimized[col] = df_optimized[col].astype(np.uint8)
                elif c_max < 65535:
                    df_optimized[col] = df_optimized[col].astype(np.uint16)
                elif c_max < 4294967295:
                    df_optimized[col] = df_optimized[col].astype(np.uint32)
            else:
                if c_min > -128 and c_max < 127:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif c_min > -32768 and c_max < 32767:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif c_min > -2147483648 and c_max < 2147483647:
                    df_optimized[col] = df_optimized[col].astype(np.int32)

        # Optimize floats
        elif df_optimized[col].dtype in ["float64"]:
            df_optimized[col] = df_optimized[col].astype(np.float32)

        # Optimize objects/strings
        elif df_optimized[col].dtype == "object":
            if (
                df_optimized[col].nunique() / len(df_optimized) < 0.5
            ):  # If less than 50% unique values
                df_optimized[col] = df_optimized[col].astype("category")

    return df_optimized
