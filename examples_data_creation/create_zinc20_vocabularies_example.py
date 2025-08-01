#!/usr/bin/env python3
"""
Example script for creating vocabularies from ZINC20 dataset using multiprocessing.

This script demonstrates how to use the enhanced create_vocabularies.py with
multiprocessing and streaming to efficiently process large datasets like ZINC20
while managing memory usage for laptops.

Usage:
    python examples/create_zinc20_vocabularies_example.py --smiles_input path/to/zinc20.smiles
    python examples/create_zinc20_vocabularies_example.py --smiles_input /export/ZINC/library_prepared/

Requirements:
    - Downloaded ZINC20 SMILES file(s) (see download instructions in script)
    - Sufficient memory (script now uses streaming for memory efficiency)
    - Multiple CPU cores for parallel processing
"""

import os
import sys
import argparse
import time
import glob
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mole.data.create_vocabularies import (
    create_guacamol_vocabularies,
    find_smiles_files,
)


def download_instructions():
    """Print instructions for downloading ZINC20 dataset."""
    print("\n" + "=" * 80)
    print("📦 ZINC20 DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 80)
    print("\n🔗 Option 1: ZINC20-ML (Machine Learning Ready)")
    print("   URL: https://files.docking.org/zinc20-ML/")
    print("   Command:")
    print("   wget https://files.docking.org/zinc20-ML/ZINC20-ML_smiles.tar.gz")
    print("   tar -xzf ZINC20-ML_smiles.tar.gz")
    print("   # This gives you 100 files with 10M molecules each (1B total)")

    print("\n🔗 Option 2: Official ZINC20 Website")
    print("   URL: https://zinc20.docking.org/")
    print("   - Browse and download specific subsets")
    print("   - Filter by properties, catalogs, etc.")

    print("\n🔗 Option 3: Pre-prepared Library (like your folder)")
    print("   Folder: /export/ZINC/library_prepared/")
    print("   - Multiple SMILES files in one folder")
    print("   - Script automatically finds and processes all files")

    print("\n📊 Expected file sizes:")
    print("   • ZINC20-ML SMILES: 8.9GB compressed")
    print("   • Individual files: ~1M-10M molecules per file typically")
    print("   • Full dataset: 230M+ purchasable compounds")

    print("\n⚡ Memory efficiency:")
    print("   • Script now uses STREAMING to avoid loading all files at once")
    print("   • Default batch size: 100,000 molecules (adjust with --batch_size)")
    print("   • Laptop-friendly: processes files one at a time")
    print("   • Memory usage: ~1-2GB typical (vs 10+ GB without streaming)")

    print("\n📁 Supported file formats:")
    print("   • .smiles, .smi, .txt, .csv files")
    print("   • Tab-separated or comma-separated (first column = SMILES)")
    print("   • Plain text (one SMILES per line)")
    print("   • Comments starting with # are ignored")
    print("=" * 80)


def estimate_processing_time(num_molecules, n_processes):
    """Estimate processing time based on benchmarks."""
    # Rough estimate: ~1000 molecules/second/process for radius 0
    molecules_per_sec_per_process = 1000
    total_throughput = molecules_per_sec_per_process * n_processes

    # Account for different radii (0, 1, 2) and functional/structural
    num_vocabularies = 6  # 3 radii × 2 feature types

    estimated_seconds = (num_molecules * num_vocabularies) / total_throughput
    estimated_minutes = estimated_seconds / 60
    estimated_hours = estimated_minutes / 60

    return estimated_seconds, estimated_minutes, estimated_hours


def estimate_molecules_in_folder(folder_path, sample_files=3):
    """Estimate total molecules by sampling a few files."""
    try:
        file_list = find_smiles_files(folder_path, verbose=False)
        if not file_list:
            return 0

        # Sample a few files to estimate
        sample_files = min(sample_files, len(file_list))
        sample_counts = []

        for file_path in file_list[:sample_files]:
            try:
                with open(file_path, "r") as f:
                    count = sum(
                        1 for line in f if line.strip() and not line.startswith("#")
                    )
                    sample_counts.append(count)
            except:
                continue

        if sample_counts:
            avg_molecules = sum(sample_counts) / len(sample_counts)
            total_estimate = int(avg_molecules * len(file_list))
            return total_estimate

    except:
        pass

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Create vocabularies from ZINC20 dataset with multiprocessing and streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single ZINC20 file
  python examples/create_zinc20_vocabularies_example.py --smiles_input zinc20_smiles.txt

  # Process folder with multiple ZINC files (MEMORY EFFICIENT)
  python examples/create_zinc20_vocabularies_example.py --smiles_input /export/ZINC/library_prepared/

  # Laptop-friendly processing with small batch size
  python examples/create_zinc20_vocabularies_example.py \\
    --smiles_input /export/ZINC/library_prepared/ \\
    --n_processes 4 \\
    --batch_size 50000

  # Test on first 100K molecules only
  python examples/create_zinc20_vocabularies_example.py \\
    --smiles_input /export/ZINC/library_prepared/ \\
    --max_molecules 100000

  # Fast processing (structural only, small batches)
  python examples/create_zinc20_vocabularies_example.py \\
    --smiles_input /export/ZINC/library_prepared/ \\
    --no_functional \\
    --batch_size 25000
        """,
    )

    parser.add_argument(
        "--smiles_input",
        type=str,
        required=True,
        help="Path to ZINC20 SMILES file or folder containing SMILES files",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="zinc20_vocabularies",
        help="Output directory for vocabulary files (default: zinc20_vocabularies)",
    )

    parser.add_argument(
        "--radii",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Morgan fingerprint radii to process (default: [0, 1, 2])",
    )

    parser.add_argument(
        "--max_molecules",
        type=int,
        default=None,
        help="Maximum number of molecules to process (for testing)",
    )

    parser.add_argument(
        "--no_functional",
        action="store_true",
        help="Skip functional environment vocabularies (process only structural)",
    )

    parser.add_argument(
        "--n_processes",
        type=int,
        default=None,
        help="Number of processes to use (default: CPU count)",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Molecules per chunk for multiprocessing (default: auto-calculated)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Molecules per batch for streaming (default: 100,000 for memory efficiency)",
    )

    parser.add_argument(
        "--download_help",
        action="store_true",
        help="Show ZINC20 download instructions and exit",
    )

    args = parser.parse_args()

    # Show download instructions if requested
    if args.download_help:
        download_instructions()
        return

    # Check if input path exists
    if not os.path.exists(args.smiles_input):
        print(f"❌ Error: Input path not found: {args.smiles_input}")
        print("\n💡 Use --download_help to see download instructions")
        return

    # Analyze input
    if os.path.isfile(args.smiles_input):
        input_type = "file"
        print(f"📄 Input: Single file - {args.smiles_input}")

        # Count lines in file
        print("📊 Counting molecules in file...")
        try:
            with open(args.smiles_input, "r") as f:
                total_molecules = sum(
                    1 for line in f if line.strip() and not line.startswith("#")
                )
        except Exception as e:
            print(f"⚠️  Could not count molecules: {e}")
            total_molecules = None

    elif os.path.isdir(args.smiles_input):
        input_type = "folder"
        print(f"📁 Input: Folder - {args.smiles_input}")

        # Find files in folder
        file_list = find_smiles_files(args.smiles_input, verbose=True)

        if len(file_list) == 0:
            print("❌ No SMILES files found in folder!")
            print("   Supported extensions: .smiles, .smi, .txt, .csv")
            return

        # Estimate total molecules (avoid loading all files)
        print("📊 Estimating total molecules by sampling files...")
        total_molecules = estimate_molecules_in_folder(args.smiles_input)

        if total_molecules:
            print(f"📊 Estimated total molecules: {total_molecules:,}")
        else:
            print("⚠️  Could not estimate molecule count")

    else:
        print(f"❌ Error: Invalid input path: {args.smiles_input}")
        return

    # Processing setup
    molecules_to_process = (
        min(total_molecules, args.max_molecules or total_molecules)
        if total_molecules
        else args.max_molecules
    )

    if args.max_molecules and total_molecules:
        print(f"🔢 Will process first: {molecules_to_process:,} molecules")

    # Processing estimation
    n_processes = args.n_processes or os.cpu_count()
    batch_size = args.batch_size or 100000

    if molecules_to_process:
        estimated_sec, estimated_min, estimated_hours = estimate_processing_time(
            molecules_to_process, n_processes
        )

        print("\n⏱️  PROCESSING ESTIMATION:")
        print(f"   • Using {n_processes} CPU cores")
        print(f"   • Batch size: {batch_size:,} molecules (memory efficient)")
        print(
            f"   • Estimated time: {estimated_hours:.1f} hours ({estimated_min:.0f} minutes)"
        )
    else:
        print("\n⏱️  PROCESSING SETUP:")
        print(f"   • Using {n_processes} CPU cores")
        print(f"   • Batch size: {batch_size:,} molecules (memory efficient)")

    # Set use_features options
    use_features_options = [False]  # Always include structural
    if not args.no_functional:
        use_features_options.append(True)  # Add functional

    vocab_types = len(use_features_options)
    vocab_count = len(args.radii) * vocab_types
    print(
        f"   • Creating {vocab_count} vocabularies ({len(args.radii)} radii × {vocab_types} types)"
    )

    # Memory efficiency note
    print(f"\n💾 MEMORY EFFICIENCY:")
    print(f"   • Streaming approach: processes {batch_size:,} molecules at a time")
    print(f"   • Memory usage: ~1-2GB typical (vs 10+ GB loading all files)")
    print(f"   • Laptop-friendly: no need to load entire dataset into RAM")

    # Confirm before processing large datasets
    if molecules_to_process and molecules_to_process > 1_000_000:  # 1M molecules
        response = input(
            f"\n⚠️  About to process {molecules_to_process:,} molecules. Continue? (y/N): "
        )
        if response.lower() != "y":
            print("❌ Cancelled by user")
            return

    print("\n🚀 Starting vocabulary creation with streaming...")
    start_time = time.time()

    try:
        # Create vocabularies
        vocab_stats = create_guacamol_vocabularies(
            smiles_input=args.smiles_input,
            output_dir=args.output_dir,
            radii=args.radii,
            use_features_options=use_features_options,
            max_molecules=args.max_molecules,
            verbose=True,
            n_processes=args.n_processes,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
        )

        # Processing complete
        end_time = time.time()
        total_time = end_time - start_time

        print("\n🎉 PROCESSING COMPLETE!")
        print(
            f"   • Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)"
        )

        if vocab_stats:
            # Calculate actual molecules processed
            actual_processed = (
                vocab_stats[list(vocab_stats.keys())[0]]["valid_molecules"]
                + vocab_stats[list(vocab_stats.keys())[0]]["invalid_molecules"]
            )
            print(f"   • Molecules processed: {actual_processed:,}")
            print(f"   • Molecules per second: {actual_processed/total_time:.0f}")

        print(f"   • Output directory: {args.output_dir}")

        # Print vocabulary summary
        print("\n📚 CREATED VOCABULARIES:")
        for vocab_name, stats in vocab_stats.items():
            print(f"   • {vocab_name}")
            print(f"     - Atom environments: {stats['total_fingerprints']:,}")
            print(f"     - File size: {stats['file_size_mb']:.1f} MB")

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        print(f"   • Partial results may be in: {args.output_dir}")

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        print("   • Check file format and try with --max_molecules 1000 for testing")
        print("   • Try reducing --batch_size if running out of memory")


if __name__ == "__main__":
    main()
