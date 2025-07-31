"""Main processing pipeline for version-based dataset processing.

AIDEV-NOTE: Complete pipeline that reads YAML configurations, loads datasets,
applies processing functions, and saves in the expected format.
"""

import logging
import yaml
from pathlib import Path
from typing import List, Optional
import pandas as pd
from datasets import Dataset, concatenate_datasets

from src.config.version_schemas import VersionConfig, DatasetConfig
from src.dataset.registry import DatasetRegistry
from src.dataset import format_converters  # Import to register processing functions

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """Main processor for version-based dataset processing."""

    def __init__(self, config_path: str):
        """Initialize processor with configuration file.

        Args:
            config_path: Path to the version YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> VersionConfig:
        """Load and validate configuration from YAML file.

        Returns:
            Validated configuration object
        """
        logger.info(f"Loading configuration from {self.config_path}")

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # Validate configuration with Pydantic
        config = VersionConfig(**config_data)
        logger.info(f"Loaded configuration for version {config.version}")

        return config

    def _load_single_dataset(self, dataset_config: DatasetConfig) -> pd.DataFrame:
        """Load a single dataset according to its configuration.

        Args:
            dataset_config: Configuration for the dataset

        Returns:
            Raw dataset as DataFrame
        """
        logger.info(f"Loading dataset: {dataset_config.name}")

        # Prepare dataset loading configuration
        load_config = {
            "dataset_id": dataset_config.dataset_id,
            "file_path": dataset_config.file_path,
            "split": dataset_config.split,
            "streaming": dataset_config.streaming,
            **dataset_config.extra_params,
        }

        # Remove None values
        load_config = {k: v for k, v in load_config.items() if v is not None}

        # Create dataset instance
        dataset = DatasetRegistry.create_dataset(dataset_config.type, load_config)

        # Load the data
        df = dataset.load()

        # Apply sample size limit if specified
        if dataset_config.sample_size is not None:
            original_size = len(df)
            df = df.head(dataset_config.sample_size)
            logger.info(f"Sampled {len(df)} rows from {original_size} total")

        logger.info(f"Loaded {len(df)} rows from {dataset_config.name}")
        return df

    def _process_single_dataset(
        self, dataset_config: DatasetConfig, raw_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Process single dataset to expected format.

        Args:
            dataset_config: Configuration for the dataset
            raw_data: Raw dataset data

        Returns:
            Processed dataset in expected format
        """
        logger.info(
            f"Processing dataset {dataset_config.name} with function {dataset_config.processing_function}"
        )

        # Apply processing function
        processed_data = DatasetRegistry.process_dataset(
            dataset_config.processing_function, raw_data, use_custom_processing=True
        )

        logger.info(f"Processed {len(processed_data)} rows for {dataset_config.name}")
        return processed_data

    def _merge_datasets(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple processed datasets.

        Args:
            datasets: List of processed DataFrames

        Returns:
            Merged DataFrame
        """
        logger.info(f"Merging {len(datasets)} datasets")

        if not datasets:
            raise ValueError("No datasets to merge")

        if len(datasets) == 1:
            merged = datasets[0]
        else:
            # Concatenate all datasets
            merged = pd.concat(datasets, ignore_index=True)

        # Apply processing configuration
        if self.config.processing.shuffle:
            logger.info("Shuffling merged dataset")
            merged = merged.sample(
                frac=1, random_state=self.config.processing.seed
            ).reset_index(drop=True)

        logger.info(f"Merged dataset has {len(merged)} total rows")
        return merged

    def _save_dataset(self, data: pd.DataFrame) -> None:
        """Save processed dataset in HuggingFace format.

        Args:
            data: Processed dataset to save
        """
        logger.info("Saving dataset in HuggingFace format")

        # Create output directory
        output_path = Path(self.config.output.path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert to HuggingFace Dataset
        hf_dataset = Dataset.from_pandas(data)

        # Add metadata
        if self.config.output.description:
            hf_dataset.info.description = self.config.output.description

        # Save to disk
        save_path = output_path / self.config.output.name
        hf_dataset.save_to_disk(str(save_path))

        logger.info(f"Dataset saved to {save_path}")

        # Save configuration alongside dataset for reference
        config_save_path = save_path / "processing_config.yaml"
        with open(config_save_path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.config.dict(), f, default_flow_style=False, allow_unicode=True
            )

        logger.info(f"Configuration saved to {config_save_path}")

    def process(self) -> None:
        """Execute the complete processing pipeline."""
        logger.info(f"Starting processing pipeline for version {self.config.version}")

        processed_datasets = []

        # Process each dataset
        for dataset_config in self.config.datasets:
            try:
                # Load raw data
                raw_data = self._load_single_dataset(dataset_config)

                # Process to expected format
                processed_data = self._process_single_dataset(dataset_config, raw_data)

                processed_datasets.append(processed_data)

            except Exception as e:
                logger.error(f"Failed to process dataset {dataset_config.name}: {e}")
                raise

        # Merge all processed datasets
        merged_data = self._merge_datasets(processed_datasets)

        # Save final dataset
        self._save_dataset(merged_data)

        logger.info("Processing pipeline completed successfully")

        # Print summary
        self._print_summary(merged_data)

    def _print_summary(self, data: pd.DataFrame) -> None:
        """Print processing summary.

        Args:
            data: Final processed dataset
        """
        print("\n" + "=" * 60)
        print(f"🎉 PROCESSING COMPLETE - Version {self.config.version}")
        print("=" * 60)

        print(f"📊 Dataset Summary:")
        print(f"  • Total rows: {len(data):,}")
        print(f"  • Columns: {list(data.columns)}")
        print(f"  • Output path: {self.config.output.path}/{self.config.output.name}")

        print(f"\n📋 Source Breakdown:")
        if "source" in data.columns:
            source_counts = data["source"].value_counts()
            for source, count in source_counts.items():
                print(f"  • {source}: {count:,} rows")

        print(f"\n⚙️  Processing Settings:")
        print(f"  • Format: {self.config.processing.output_format}")
        print(f"  • Shuffled: {self.config.processing.shuffle}")
        print(f"  • Seed: {self.config.processing.seed}")

        print("\n✅ Dataset ready for training!")


def main():
    """CLI entry point for dataset processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process datasets according to version configuration"
    )
    parser.add_argument("config", help="Path to version configuration YAML file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run processing
    processor = DatasetProcessor(args.config)
    processor.process()


if __name__ == "__main__":
    main()
