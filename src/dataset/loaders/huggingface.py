"""HuggingFace dataset loader implementation.

AIDEV-NOTE: Dataset loader for HuggingFace Hub datasets.
Supports both public and private datasets with authentication.
"""

import logging
from typing import Any

import pandas as pd
from datasets import load_dataset

from ..base import BaseDataset
from ..registry import DatasetRegistry

logger = logging.getLogger(__name__)


@DatasetRegistry.register("huggingface")
class HuggingFaceDataset(BaseDataset):
    """Dataset loader for HuggingFace Hub datasets.

    AIDEV-NOTE: Supports streaming, authentication, and various HF dataset formats.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize HuggingFace dataset loader.

        Args:
            config: Configuration with keys:
                - dataset_id: HuggingFace dataset identifier
                - split: Dataset split to load (optional, defaults to 'train')
                - streaming: Whether to use streaming (optional, defaults to False)
                - token: HuggingFace token for private datasets (optional)
                - trust_remote_code: Whether to trust remote code (optional)
        """
        super().__init__(config)

        self.dataset_id = config.get("dataset_id")
        if not self.dataset_id:
            raise ValueError("dataset_id is required for HuggingFace datasets")

        self.split = config.get("split", "train")
        self.streaming = config.get("streaming", False)
        self.token = config.get("token")
        self.trust_remote_code = config.get("trust_remote_code", False)

        logger.info(f"Initialized HuggingFace dataset: {self.dataset_id}")

    def load(self) -> pd.DataFrame:
        """Load the HuggingFace dataset.

        Returns:
            DataFrame with the loaded data

        Raises:
            Exception: If dataset loading fails
        """
        try:
            logger.info(
                f"Loading HuggingFace dataset: {self.dataset_id}, split: {self.split}"
            )

            # Load dataset from HuggingFace Hub
            dataset = load_dataset(
                self.dataset_id,
                split=self.split,
                streaming=self.streaming,
                token=self.token,
                trust_remote_code=self.trust_remote_code,
            )

            # Convert to pandas DataFrame
            if self.streaming:
                # For streaming datasets, we need to consume some data
                # This is a simplified approach - in practice you might want to batch process
                data_list = []
                for i, item in enumerate(dataset):
                    data_list.append(item)
                    # Limit to prevent memory issues with streaming
                    if i >= 10000:  # Configurable limit
                        break
                df = pd.DataFrame(data_list)
            else:
                df = dataset.to_pandas()

            logger.info(f"Loaded {len(df)} rows from HuggingFace dataset")
            return df

        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset {self.dataset_id}: {e}")
            raise

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate the loaded HuggingFace dataset.

        Args:
            data: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        if data.empty:
            logger.error("Dataset is empty")
            return False

        # Basic validation - ensure we have some data
        if len(data) == 0:
            logger.error("Dataset has no rows")
            return False

        # Check for required columns (this would be dataset-specific)
        # For now, just check that we have at least one column
        if len(data.columns) == 0:
            logger.error("Dataset has no columns")
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
        return f"huggingface:{self.dataset_id}"

    def get_info(self) -> dict[str, Any]:
        """Get extended information about the dataset.

        Returns:
            Dictionary with dataset metadata
        """
        info = super().get_info()
        info.update(
            {
                "dataset_id": self.dataset_id,
                "split": self.split,
                "streaming": self.streaming,
                "source_type": "huggingface",
            }
        )
        return info
