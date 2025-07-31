"""CLI for running dataset processing pipeline.

AIDEV-NOTE: Simple command-line interface for the dataset processing flow.
Usage: python run_pipeline.py src/config/versions/ver_1.1.yaml
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Import the pipeline
from src.core.pipeline import run_pipeline_from_yaml


def main():
    """Main CLI entry point."""
    if len(sys.argv) != 2:
        print("❌ Usage: python run_pipeline.py <version_yaml_file>")
        print("📄 Example: python run_pipeline.py src/config/versions/ver_1.1.yaml")
        sys.exit(1)

    version_file = sys.argv[1]

    # Check if file exists
    if not Path(version_file).exists():
        print(f"❌ Error: Version file not found: {version_file}")
        sys.exit(1)

    try:
        run_pipeline_from_yaml(version_file)
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
