"""Local files dataset loader implementation.

AIDEV-NOTE: Dataset loader for local files (CSV, JSON, Parquet, etc.).
Supports various file formats and automatic format detection.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ..base import BaseDataset
from ..registry import DatasetRegistry

logger = logging.getLogger(__name__)


@DatasetRegistry.register("local_files")
class LocalFilesDataset(BaseDataset):
    """Dataset loader for local files.

    AIDEV-NOTE: Supports CSV, JSON, Parquet, and other pandas-readable formats.
    """

    SUPPORTED_FORMATS = {
        ".csv": "csv",
        ".json": "json",
        ".jsonl": "jsonl",
        ".parquet": "parquet",
        ".xlsx": "excel",
        ".tsv": "tsv",
    }

    def __init__(self, config: dict[str, Any]):
        """Initialize local files dataset loader.

        Args:
            config: Configuration with keys:
                - file_path: Path to the file or directory
                - format: File format (optional, auto-detected if not provided)
                - encoding: File encoding (optional, defaults to 'utf-8')
                - sep: Separator for CSV files (optional, defaults to ',')
                - **kwargs: Additional pandas read options
        """
        super().__init__(config)

        self.file_path = config.get("file_path")
        if not self.file_path:
            raise ValueError("file_path is required for local files dataset")

        self.file_path = Path(self.file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self.format = config.get("format")
        self.encoding = config.get("encoding", "utf-8")
        self.sep = config.get("sep", ",")

        # Extract additional pandas options
        excluded_params = [
            "file_path",
            "format",
            "encoding",
            "sep",
            "dataset_id",
            "split",
            "streaming",  # Exclude pipeline-specific params
        ]
        self.pandas_options = {
            k: v for k, v in config.items() if k not in excluded_params
        }

        # Auto-detect format if not provided
        if not self.format:
            self.format = self._detect_format()

        logger.info(
            f"Initialized local files dataset: {self.file_path} (format: {self.format})"
        )

    def _detect_format(self) -> str:
        """Auto-detect file format from extension.

        Returns:
            Detected format string

        Raises:
            ValueError: If format is not supported
        """
        suffix = self.file_path.suffix.lower()

        if suffix in self.SUPPORTED_FORMATS:
            return self.SUPPORTED_FORMATS[suffix]

        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: {list(self.SUPPORTED_FORMATS.keys())}"
        )

    def load(self) -> pd.DataFrame:
        """Load the local file dataset.

        Returns:
            DataFrame with the loaded data

        Raises:
            Exception: If file loading fails
        """
        try:
            logger.info(f"Loading local file: {self.file_path} (format: {self.format})")

            if self.format == "csv":
                df = pd.read_csv(
                    self.file_path,
                    encoding=self.encoding,
                    sep=self.sep,
                    **self.pandas_options,
                )
            elif self.format == "tsv":
                df = pd.read_csv(
                    self.file_path,
                    encoding=self.encoding,
                    sep="\t",
                    **self.pandas_options,
                )
            elif self.format == "json":
                df = pd.read_json(
                    self.file_path, encoding=self.encoding, **self.pandas_options
                )
            elif self.format == "jsonl":
                df = pd.read_json(
                    self.file_path,
                    lines=True,
                    encoding=self.encoding,
                    **self.pandas_options,
                )
            elif self.format == "parquet":
                df = pd.read_parquet(self.file_path, **self.pandas_options)
            elif self.format == "excel":
                df = pd.read_excel(self.file_path, **self.pandas_options)
            else:
                raise ValueError(f"Unsupported format: {self.format}")

            logger.info(f"Loaded {len(df)} rows from local file")
            return df

        except Exception as e:
            logger.error(f"Failed to load local file {self.file_path}: {e}")
            raise

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate the loaded local file dataset.

        Args:
            data: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        if data.empty:
            logger.error("Dataset is empty")
            return False

        # Basic validation
        if len(data) == 0:
            logger.error("Dataset has no rows")
            return False

        if len(data.columns) == 0:
            logger.error("Dataset has no columns")
            return False

        # Check for reasonable file size
        file_size = self.file_path.stat().st_size
        if file_size == 0:
            logger.error("File is empty")
            return False

        logger.info(
            f"Dataset validation passed: {len(data)} rows, {len(data.columns)} columns"
        )
        return True

    @property
    def source_name(self) -> str:
        """Return the source name for this dataset.

        Returns:
            Source name string
        """
        return f"local_files:{self.file_path.name}"

    def get_info(self) -> dict[str, Any]:
        """Get extended information about the dataset.

        Returns:
            Dictionary with dataset metadata
        """
        info = super().get_info()

        # Get file stats
        stat = self.file_path.stat()

        info.update(
            {
                "file_path": str(self.file_path),
                "format": self.format,
                "encoding": self.encoding,
                "file_size": stat.st_size,
                "file_modified": stat.st_mtime,
                "source_type": "local_files",
            }
        )
        return info
