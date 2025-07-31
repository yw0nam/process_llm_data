"""Main processing pipeline for the dataset processing flow.

AIDEV-NOTE: Implements the complete flow:
1. Read version YAML file
2. Register datasets and processing functions
3. Process datasets to expected input format
4. Merge and save as HuggingFace dataset
"""

import logging
import yaml
from pathlib import Path
from typing import List, Optional
import pandas as pd
from datasets import Dataset, concatenate_datasets

from src.config.version_schemas import VersionConfig
from src.dataset.registry import DatasetRegistry
from src.dataset.format_converters import *  # This registers all processing functions

logger = logging.getLogger(__name__)


class DatasetProcessingPipeline:
    """Main pipeline for processing datasets according to version configuration."""

    def __init__(self, version_file: Path):
        """Initialize pipeline with version configuration.

        Args:
            version_file: Path to version YAML file
        """
        self.version_file = Path(version_file)
        self.config = self._load_config()

        logger.info(f"Initialized pipeline for version {self.config.version}")

    def _load_config(self) -> VersionConfig:
        """Load and validate version configuration from YAML file."""
        if not self.version_file.exists():
            raise FileNotFoundError(f"Version file not found: {self.version_file}")

        with open(self.version_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        return VersionConfig(**config_data)

    def _create_dataset_loader_config(self, dataset_config) -> dict:
        """Create loader configuration from dataset config."""
        loader_config = {
            "dataset_id": dataset_config.dataset_id,
            "split": dataset_config.split,
            "streaming": dataset_config.streaming or False,
        }

        if dataset_config.file_path:
            loader_config["file_path"] = dataset_config.file_path

        # Add any extra parameters
        if dataset_config.extra_params:
            loader_config.update(dataset_config.extra_params)

        return loader_config

    def _load_and_sample_dataset(self, dataset_config) -> pd.DataFrame:
        """Load dataset and apply sampling if specified."""
        logger.info(f"Loading dataset: {dataset_config.name}")

        # Create dataset loader configuration
        loader_config = self._create_dataset_loader_config(dataset_config)

        # Create dataset instance
        if dataset_config.type == "huggingface":
            dataset = DatasetRegistry.create_dataset("huggingface", loader_config)
        elif dataset_config.type == "local_files":
            dataset = DatasetRegistry.create_dataset("local_files", loader_config)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_config.type}")

        # Load the data
        df = dataset.load()

        # Apply sampling if specified
        if dataset_config.sample_size and len(df) > dataset_config.sample_size:
            df = df.sample(
                n=dataset_config.sample_size, random_state=self.config.processing.seed
            )
            logger.info(
                f"Sampled {dataset_config.sample_size} rows from {dataset_config.name}"
            )

        logger.info(f"Loaded {len(df)} rows from {dataset_config.name}")
        return df

    def _process_dataset_to_format(
        self, df: pd.DataFrame, dataset_config
    ) -> pd.DataFrame:
        """Process dataset using its registered processing function."""
        logger.info(
            f"Processing dataset {dataset_config.name} with function {dataset_config.processing_function}"
        )

        # Process dataset using registered processing function
        processed_df = DatasetRegistry.process_dataset(
            dataset_config.processing_function, df, use_custom_processing=True
        )

        logger.info(f"Processed {len(processed_df)} rows from {dataset_config.name}")
        return processed_df

    def _merge_datasets(self, processed_datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge all processed datasets according to merge strategy."""
        logger.info(
            f"Merging {len(processed_datasets)} datasets using {self.config.processing.merge_strategy}"
        )

        if self.config.processing.merge_strategy == "concatenate":
            merged_df = pd.concat(processed_datasets, ignore_index=True)
        else:
            raise ValueError(
                f"Unsupported merge strategy: {self.config.processing.merge_strategy}"
            )

        # Shuffle if requested
        if self.config.processing.shuffle:
            merged_df = merged_df.sample(
                frac=1, random_state=self.config.processing.seed
            ).reset_index(drop=True)
            logger.info("Shuffled merged dataset")

        logger.info(f"Final merged dataset has {len(merged_df)} rows")
        return merged_df

    def _save_dataset(self, df: pd.DataFrame) -> None:
        """Save the final dataset in HuggingFace format."""
        logger.info(f"Saving dataset to {self.config.output.path}")

        # Create output directory
        output_path = Path(self.config.output.path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert to HuggingFace Dataset
        dataset = Dataset.from_pandas(df)

        # Add metadata if provided
        if self.config.output.description:
            dataset.info.description = self.config.output.description
        if self.config.output.version:
            dataset.info.version = self.config.output.version

        # Save dataset
        full_output_path = output_path / self.config.output.name
        dataset.save_to_disk(str(full_output_path))

        logger.info(f"Successfully saved dataset to {full_output_path}")

        # Save metadata
        metadata = {
            "version": self.config.version,
            "description": self.config.description,
            "num_samples": len(df),
            "datasets_used": [d.name for d in self.config.datasets],
            "processing_functions": [
                d.processing_function for d in self.config.datasets
            ],
            "output_format": self.config.processing.output_format,
        }

        metadata_path = output_path / f"{self.config.output.name}_metadata.yaml"
        with open(metadata_path, "w", encoding="utf-8") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        logger.info(f"Saved metadata to {metadata_path}")

    def run(self) -> None:
        """Execute the complete processing pipeline."""
        logger.info(f"Starting pipeline execution for version {self.config.version}")

        processed_datasets = []

        # Step 1-3: Process each dataset
        for dataset_config in self.config.datasets:
            try:
                # Load and sample dataset
                df = self._load_and_sample_dataset(dataset_config)

                # Process to expected format
                processed_df = self._process_dataset_to_format(df, dataset_config)

                processed_datasets.append(processed_df)

            except Exception as e:
                logger.error(f"Failed to process dataset {dataset_config.name}: {e}")
                raise

        # Step 4: Merge and save
        if not processed_datasets:
            raise ValueError("No datasets were successfully processed")

        merged_df = self._merge_datasets(processed_datasets)
        self._save_dataset(merged_df)

        logger.info("Pipeline execution completed successfully!")

    def get_summary(self) -> dict:
        """Get a summary of the pipeline configuration."""
        return {
            "version": self.config.version,
            "description": self.config.description,
            "num_datasets": len(self.config.datasets),
            "datasets": [
                {
                    "name": d.name,
                    "type": d.type,
                    "sample_size": d.sample_size,
                    "processing_function": d.processing_function,
                }
                for d in self.config.datasets
            ],
            "output_path": str(self.config.output.path),
            "output_name": self.config.output.name,
        }


def run_pipeline_from_yaml(version_file: str) -> None:
    """Convenience function to run pipeline from YAML file.

    Args:
        version_file: Path to version YAML file
    """
    pipeline = DatasetProcessingPipeline(version_file)

    # Print summary
    summary = pipeline.get_summary()
    print(f"🚀 Starting Pipeline: {summary['version']}")
    print(f"📄 Description: {summary['description']}")
    print(f"📊 Datasets: {summary['num_datasets']}")
    for dataset in summary["datasets"]:
        print(
            f"  - {dataset['name']}: {dataset['sample_size']} samples ({dataset['processing_function']})"
        )
    print(f"💾 Output: {summary['output_path']}/{summary['output_name']}")
    print("-" * 60)

    # Run pipeline
    pipeline.run()

    print("✅ Pipeline completed successfully!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python pipeline.py <version_yaml_file>")
        sys.exit(1)

    version_file = sys.argv[1]
    run_pipeline_from_yaml(version_file)
