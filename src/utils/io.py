"""I/O utilities for file operations.

AIDEV-NOTE: Standardized file I/O operations with proper error handling.
Enhanced to focus on Hugging Face datasets output only.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

logger = logging.getLogger(__name__)


def read_json(file_path: str | Path) -> dict[str, Any] | list[Any]:
    """Read JSON file safely.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def write_json(
    data: dict[str, Any] | list[Any], file_path: str | Path, indent: int = 2
) -> None:
    """Write data to JSON file safely.

    Args:
        data: Data to write
        file_path: Output file path
        indent: JSON indentation level
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_jsonl(file_path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of parsed JSON objects
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    data = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(data: list[dict[str, Any]], file_path: str | Path) -> None:
    """Write data to JSONL file.

    Args:
        data: List of dictionaries to write
        file_path: Output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(str(item) + "\n")


def save_dataset(
    dataset: Dataset | DatasetDict,
    output_path: str | Path,
    format: str = "datasets",
) -> None:
    """Save dataset in specified format.

    Args:
        dataset: Dataset to save
        output_path: Output directory path
        format: Save format ("datasets", "json", "parquet")
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if format == "datasets":
        dataset.save_to_disk(str(output_path))
    elif format == "json":
        if isinstance(dataset, DatasetDict):
            for split_name, split_dataset in dataset.items():
                split_path = output_path / f"{split_name}.json"
                split_dataset.to_json(str(split_path))
        else:
            dataset.to_json(str(output_path / "data.json"))
    elif format == "parquet":
        if isinstance(dataset, DatasetDict):
            for split_name, split_dataset in dataset.items():
                split_path = output_path / f"{split_name}.parquet"
                split_dataset.to_parquet(str(split_path))
        else:
            dataset.to_parquet(str(output_path / "data.parquet"))
    else:
        raise ValueError(f"Unsupported format: {format}")


def ensure_directory(path: str | Path) -> Path:
    """Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class DatasetSaver:
    """AIDEV-NOTE: Enhanced dataset saver focused on HF datasets only."""

    def __init__(self, output_dir: str, dataset_name: str | None = None):
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_with_splits(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        push_to_hub: bool = False,
        hub_repo_id: str | None = None,
        hub_private: bool = True,
    ) -> str:
        """
        Save DataFrame as HF dataset with automatic train/val/test splits.

        Args:
            data: DataFrame to save
            train_ratio: Ratio for training split
            val_ratio: Ratio for validation split
            test_ratio: Ratio for test split
            push_to_hub: Whether to push to HF Hub
            hub_repo_id: Repository ID for HF Hub
            hub_private: Whether to make repo private

        Returns:
            Path to saved dataset
        """
        # Create splits
        dataset_dict = self._create_splits(data, train_ratio, val_ratio, test_ratio)

        # Convert to HF dataset
        hf_dataset = DatasetDict(
            {
                split: Dataset.from_pandas(df, preserve_index=False)
                for split, df in dataset_dict.items()
            }
        )

        # Determine save path
        save_path = self.output_dir
        if self.dataset_name:
            save_path = save_path / self.dataset_name

        # Save to disk
        hf_dataset.save_to_disk(str(save_path))
        logger.info(f"Dataset saved to {save_path}")

        # Push to hub if requested
        if push_to_hub and hub_repo_id:
            try:
                hf_dataset.push_to_hub(hub_repo_id, private=hub_private)
                logger.info(f"Dataset pushed to hub: {hub_repo_id}")
            except Exception as e:
                logger.error(f"Failed to push to hub: {e}")

        return str(save_path)

    def _create_splits(
        self,
        data: pd.DataFrame,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> dict[str, pd.DataFrame]:
        """Create train/val/test splits."""
        total_size = len(data)

        # Calculate split sizes
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size

        # Shuffle data
        data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

        splits = {}
        splits["train"] = data_shuffled[:train_size]

        if val_size > 0:
            splits["validation"] = data_shuffled[train_size : train_size + val_size]

        if test_size > 0:
            splits["test"] = data_shuffled[train_size + val_size :]

        logger.info(f"Created splits: {[(k, len(v)) for k, v in splits.items()]}")
        return splits


def load_hf_dataset(dataset_path: str | Path) -> DatasetDict:
    """Load HF dataset from local path or hub."""
    try:
        if Path(dataset_path).exists():
            return load_from_disk(str(dataset_path))
        else:
            return load_dataset(str(dataset_path))
    except Exception as e:
        logger.error(f"Failed to load HF dataset from {dataset_path}: {e}")
        raise
