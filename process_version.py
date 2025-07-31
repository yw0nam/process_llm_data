"""CLI script for running the dataset processing pipeline.

AIDEV-NOTE: Command-line interface for processing datasets according to YAML configurations.
Supports the complete flow: load -> process -> merge -> save.
"""

#!/usr/bin/env python3

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.core.dataset_processor import DatasetProcessor


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process datasets according to version configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_version.py src/config/versions/instruction_ver_2_2.yaml
  python process_version.py configs/preference_ver_1_0.yaml --log-level DEBUG
        """,
    )

    parser.add_argument("config", help="Path to version configuration YAML file")

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("processing.log"),
        ],
    )

    logger = logging.getLogger(__name__)

    try:
        # Initialize processor
        processor = DatasetProcessor(args.config)

        if args.dry_run:
            # Show configuration without processing
            print("\n🔍 DRY RUN - Configuration Preview")
            print("=" * 50)
            print(f"Version: {processor.config.version}")
            print(f"Description: {processor.config.description}")
            print(f"\nDatasets to process:")
            for ds in processor.config.datasets:
                print(f"  • {ds.name} ({ds.type}) - {ds.sample_size or 'all'} samples")
                print(f"    Processing: {ds.processing_function}")
            print(
                f"\nOutput: {processor.config.output.path}/{processor.config.output.name}"
            )
            print("\nUse without --dry-run to actually process the datasets.")
        else:
            # Run actual processing
            processor.process()

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
