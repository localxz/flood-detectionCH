#!/usr/bin/env python3
"""
Script to copy PNG tile files based on labels from a reference folder.
1. Scans reference folder to get tile numbers (labels)
2. Copies files with matching labels from source folder to destination folder
"""

import os
import shutil
import re
import sys
from pathlib import Path


def extract_tile_numbers(reference_folder):

    ref_path = Path(reference_folder)

    if not ref_path.exists():
        print(f"Error: Reference folder '{reference_folder}' does not exist.")
        return set()

    tile_pattern = re.compile(r'^tile_(\d+)\.png$', re.IGNORECASE)
    tile_numbers = set()

    for file_path in ref_path.iterdir():
        if file_path.is_file():
            match = tile_pattern.match(file_path.name)
            if match:
                tile_number = match.group(1)
                tile_numbers.add(tile_number)

    return tile_numbers


def copy_matching_tiles(source_folder, destination_folder, tile_numbers):

    source_path = Path(source_folder)
    dest_path = Path(destination_folder)

    if not source_path.exists():
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return 0, 0

    dest_path.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    not_found_count = 0

    print(f"Looking for {len(tile_numbers)} specific tiles...")

    for tile_num in sorted(tile_numbers):
        possible_names = [
            f"tile_{tile_num}.png",
            f"TILE_{tile_num}.PNG",
            f"Tile_{tile_num}.png",
            f"tile_{tile_num}.PNG"
        ]

        file_found = False
        for filename in possible_names:
            source_file = source_path / filename
            if source_file.exists():
                try:
                    dest_file = dest_path / filename
                    shutil.copy2(source_file, dest_file)
                    print(f"✓ Copied: {filename}")
                    copied_count += 1
                    file_found = True
                    break
                except Exception as e:
                    print(f"✗ Failed to copy {filename}: {e}")

        if not file_found:
            print(f"✗ Not found: tile_{tile_num}.png")
            not_found_count += 1

    return copied_count, not_found_count


def main():

    if len(sys.argv) == 4:
        reference_folder = sys.argv[1]
        source_folder = sys.argv[2]
        destination_folder = sys.argv[3]
    else:
        print("Selective PNG Tile Copy Script")
        print("=" * 40)
        print("This script will:")
        print("1. Scan a reference folder to get tile numbers")
        print("2. Copy matching tiles from source to destination")
        print()

        reference_folder = input("Enter reference folder path (contains example tiles): ").strip()
        if not reference_folder:
            print("Error: Reference folder path cannot be empty.")
            return

        source_folder = input("Enter source folder path (contains all tiles): ").strip()
        if not source_folder:
            print("Error: Source folder path cannot be empty.")
            return

        destination_folder = input("Enter destination folder path: ").strip()
        if not destination_folder:
            print("Error: Destination folder path cannot be empty.")
            return

    print(f"\nReference folder: {Path(reference_folder).absolute()}")
    print(f"Source folder: {Path(source_folder).absolute()}")
    print(f"Destination folder: {Path(destination_folder).absolute()}")
    print()

    print("Step 1: Extracting tile numbers from reference folder...")
    tile_numbers = extract_tile_numbers(reference_folder)

    if not tile_numbers:
        print("No tile files found in reference folder!")
        return

    print(f"Found {len(tile_numbers)} unique tile numbers: {sorted(tile_numbers)}")
    print()

    print("Step 2: Copying matching tiles from source folder...")
    copied, not_found = copy_matching_tiles(source_folder, destination_folder, tile_numbers)

    print(f"\nOperation complete!")
    print(f"Tiles copied: {copied}")
    print(f"Tiles not found: {not_found}")
    print(
        f"Success rate: {copied}/{len(tile_numbers)} ({100 * copied / len(tile_numbers) if tile_numbers else 0:.1f}%)")


if __name__ == "__main__":
    main()
    