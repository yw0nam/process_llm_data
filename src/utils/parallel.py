"""Parallel processing utilities for efficient data processing.

AIDEV-NOTE: Utilities for parallel processing of datasets to improve performance.
Supports both multiprocessing and concurrent.futures for different use cases.
"""

import logging
import multiprocessing as mp
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """AIDEV-NOTE: Unified parallel processing class supporting multiple backends."""

    def __init__(
        self,
        n_workers: int | None = None,
        backend: str = "process",
        chunk_size: int | None = None,
    ):
        """
        Initialize parallel processor.

        Args:
            n_workers: Number of workers (defaults to CPU count)
            backend: "process" or "thread"
            chunk_size: Size of chunks for processing
        """
        self.n_workers = n_workers or mp.cpu_count()
        self.backend = backend
        self.chunk_size = chunk_size or max(1, 1000 // self.n_workers)

        logger.info(f"Initialized {backend} pool with {self.n_workers} workers")

    def process_dataframe(
        self, df: pd.DataFrame, process_func: Callable, **kwargs
    ) -> pd.DataFrame:
        """
        Process DataFrame in parallel chunks.

        Args:
            df: DataFrame to process
            process_func: Function to apply to each chunk
            **kwargs: Additional arguments for process_func

        Returns:
            Processed DataFrame
        """
        if len(df) < self.chunk_size:
            # Small dataset, process directly
            return process_func(df, **kwargs)

        # Split into chunks
        chunks = self._split_dataframe(df, self.chunk_size)

        # Process chunks in parallel
        processed_chunks = self.map(
            partial(process_func, **kwargs), chunks, desc="Processing chunks"
        )

        # Combine results
        return pd.concat(processed_chunks, ignore_index=True)

    def process_items(
        self, items: list[Any], process_func: Callable, **kwargs
    ) -> list[Any]:
        """
        Process list of items in parallel.

        Args:
            items: List of items to process
            process_func: Function to apply to each item
            **kwargs: Additional arguments for process_func

        Returns:
            List of processed items
        """
        return self.map(partial(process_func, **kwargs), items, desc="Processing items")

    def map(
        self, func: Callable, iterable: Iterable, desc: str = "Processing", **kwargs
    ) -> list[Any]:
        """
        Map function over iterable in parallel.

        Args:
            func: Function to apply
            iterable: Items to process
            desc: Description for progress bar
            **kwargs: Additional arguments

        Returns:
            List of results
        """
        items = list(iterable)

        if len(items) == 0:
            return []

        if len(items) == 1 or self.n_workers == 1:
            # Single item or single worker, process directly
            return [func(item) for item in tqdm(items, desc=desc)]

        executor_class = (
            ProcessPoolExecutor if self.backend == "process" else ThreadPoolExecutor
        )

        results = []
        with executor_class(max_workers=self.n_workers) as executor:
            # Submit all tasks
            future_to_item = {executor.submit(func, item): item for item in items}

            # Collect results with progress bar
            with tqdm(total=len(items), desc=desc) as pbar:
                for future in as_completed(future_to_item):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        item = future_to_item[future]
                        logger.error(f"Error processing item {item}: {e}")
                        results.append(None)  # or handle error differently
                    finally:
                        pbar.update(1)

        return results

    def _split_dataframe(self, df: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
        """Split DataFrame into chunks."""
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i : i + chunk_size].copy()
            chunks.append(chunk)
        return chunks


def process_in_batches(
    data: list | pd.DataFrame,
    process_func: Callable,
    batch_size: int = 1000,
    n_workers: int = None,
    desc: str = "Processing batches",
) -> list | pd.DataFrame:
    """
    Process data in batches for memory efficiency.

    Args:
        data: Data to process (List or DataFrame)
        process_func: Function to apply to each batch
        batch_size: Size of each batch
        n_workers: Number of workers for parallel processing
        desc: Description for progress bar

    Returns:
        Processed data in same format as input
    """
    if isinstance(data, pd.DataFrame):
        # Process DataFrame in batches
        batches = [
            data.iloc[i : i + batch_size] for i in range(0, len(data), batch_size)
        ]

        if n_workers and n_workers > 1:
            processor = ParallelProcessor(n_workers=n_workers)
            processed_batches = processor.map(process_func, batches, desc=desc)
        else:
            processed_batches = [
                process_func(batch) for batch in tqdm(batches, desc=desc)
            ]

        return pd.concat(processed_batches, ignore_index=True)

    elif isinstance(data, list):
        # Process list in batches
        batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

        if n_workers and n_workers > 1:
            processor = ParallelProcessor(n_workers=n_workers)
            processed_batches = processor.map(process_func, batches, desc=desc)
        else:
            processed_batches = [
                process_func(batch) for batch in tqdm(batches, desc=desc)
            ]

        # Flatten results
        result = []
        for batch_result in processed_batches:
            if isinstance(batch_result, list):
                result.extend(batch_result)
            else:
                result.append(batch_result)
        return result

    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def safe_parallel_map(
    func: Callable,
    items: list[Any],
    n_workers: int = None,
    max_retries: int = 3,
    desc: str = "Processing",
) -> list[Any]:
    """
    Parallel map with error handling and retries.

    Args:
        func: Function to apply
        items: Items to process
        n_workers: Number of workers
        max_retries: Maximum retries for failed items
        desc: Description for progress bar

    Returns:
        List of results (None for failed items)
    """
    processor = ParallelProcessor(n_workers=n_workers)

    def safe_func(item):
        """Wrapper function with error handling."""
        for attempt in range(max_retries + 1):
            try:
                return func(item)
            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        f"Failed to process item after {max_retries + 1} attempts: {e}"
                    )
                    return None
                else:
                    logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
        return None

    return processor.map(safe_func, items, desc=desc)
