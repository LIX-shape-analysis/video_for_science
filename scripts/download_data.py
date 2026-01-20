#!/usr/bin/env python3
"""
Script to download The Well turbulent_radiative_layer_2D dataset.

Usage:
    python scripts/download_data.py --output_dir ./datasets

This will download both training and validation splits.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_well_dataset(
    output_dir: str,
    dataset_name: str = "turbulent_radiative_layer_2D",
    splits: list = None,
):
    """
    Download The Well dataset.
    
    Args:
        output_dir: Directory to store the dataset
        dataset_name: Name of the dataset to download
        splits: List of splits to download
    """
    try:
        from the_well.utils.download import well_download
    except ImportError:
        print("Error: the_well package not installed.")
        print("Please install it with: pip install the_well[benchmark]")
        sys.exit(1)
    
    if splits is None:
        splits = ["train", "valid"]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {dataset_name} dataset to {output_dir}")
    print(f"Splits: {splits}")
    print("-" * 50)
    
    for split in splits:
        print(f"\nDownloading {split} split...")
        try:
            well_download(
                base_path=output_dir,
                dataset=dataset_name,
                split=split,
            )
            print(f"Successfully downloaded {split} split")
        except Exception as e:
            print(f"Error downloading {split} split: {e}")
            continue
    
    print("\n" + "=" * 50)
    print("Download complete!")
    print(f"Dataset location: {output_dir}/{dataset_name}")
    
    # Print dataset information
    print("\n" + "=" * 50)
    print("Dataset Information:")
    print("-" * 50)
    print(f"Dataset: {dataset_name}")
    print("Fields: density, pressure, velocity_x, velocity_y (4 channels)")
    print("Spatial resolution: 128 x 384")
    print("This dataset simulates turbulent radiative layer dynamics")
    print("\nFor more information, see:")
    print("https://polymathic-ai.org/the_well/datasets/turbulent_radiative_layer_2D/")


def main():
    parser = argparse.ArgumentParser(
        description="Download The Well turbulent_radiative_layer_2D dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets",
        help="Directory to store the dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="turbulent_radiative_layer_2D",
        help="Dataset name to download"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "valid"],
        help="Splits to download (train, valid, test)"
    )
    
    args = parser.parse_args()
    
    download_well_dataset(
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        splits=args.splits,
    )


if __name__ == "__main__":
    main()
