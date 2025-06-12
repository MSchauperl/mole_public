import os
import pickle
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial
import math
import glob


def find_smiles_files(folder_path: str, verbose: bool = True):
    """
    Find all SMILES files in a folder without loading them.

    Parameters:
    -----------
    folder_path : str
        Path to folder containing SMILES files
    verbose : bool
        Print progress information

    Returns:
    --------
    list : List of file paths
    """
    if verbose:
        print(f"📂 Scanning folder: {folder_path}")

    # Find all potential SMILES files
    extensions = ["*.smiles", "*.smi", "*.txt", "*.csv"]
    all_files = []

    for ext in extensions:
        pattern = os.path.join(folder_path, ext)
        all_files.extend(glob.glob(pattern))

        # Also check subdirectories
        pattern = os.path.join(folder_path, "**", ext)
        all_files.extend(glob.glob(pattern, recursive=True))

    # Remove duplicates and sort
    all_files = sorted(list(set(all_files)))

    if verbose:
        print(f"📋 Found {len(all_files)} potential SMILES files")
        if len(all_files) <= 10:
            for f in all_files:
                print(f"   • {os.path.basename(f)}")
        else:
            for f in all_files[:5]:
                print(f"   • {os.path.basename(f)}")
            print(f"   • ... and {len(all_files)-5} more files")

    return all_files


# def stream_smiles_from_files(
#     file_list: list, max_molecules: int = None, verbose: bool = True
# ):
#     """
#     Stream SMILES from multiple files without loading all into memory.

#     Parameters:
#     -----------
#     file_list : list
#         List of file paths to process
#     max_molecules : int, optional
#         Maximum number of molecules to yield
#     verbose : bool
#         Print progress information

#     Yields:
#     -------
#     str : SMILES string
#     """
#     molecules_yielded = 0

#     for file_path in file_list:
#         if max_molecules and molecules_yielded >= max_molecules:
#             break

#         try:
#             with open(file_path, "r") as f:
#                 file_molecules = 0
#                 for line in f:
#                     line = line.strip()
#                     if line and not line.startswith(
#                         "#"
#                     ):  # Skip empty lines and comments
#                         # Handle different file formats
#                         if "\t" in line or "," in line:
#                             # Take first column if tab/comma separated
#                             smiles = line.split("\t")[0].split(",")[0].strip()
#                         else:
#                             smiles = line

#                         if smiles:
#                             yield smiles
#                             molecules_yielded += 1
#                             file_molecules += 1

#                             if max_molecules and molecules_yielded >= max_molecules:
#                                 break

#                 if verbose and file_molecules > 0:
#                     print(
#                         f"   ✓ {os.path.basename(file_path)}: {file_molecules:,} molecules"
#                     )

#         except Exception as e:
#             if verbose:
#                 print(f"   ⚠️  Could not read {os.path.basename(file_path)}: {e}")
#             continue


def get_smiles_stream(input_path: str, max_molecules: int = None, verbose: bool = True):
    """
    Get SMILES stream from either a file or folder.

    Parameters:
    -----------
    input_path : str
        Path to SMILES file or folder containing SMILES files
    max_molecules : int, optional
        Maximum number of molecules to process
    verbose : bool
        Print progress information

    Returns:
    --------
    generator : Generator yielding SMILES strings
    """
    if os.path.isfile(input_path):
        # Single file
        if verbose:
            print(f"📄 Reading SMILES from file: {input_path}")

        def file_generator():
            molecules_yielded = 0
            with open(input_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Handle different file formats
                        if "\t" in line or "," in line:
                            smiles = line.split("\t")[0].split(",")[0].strip()
                        else:
                            smiles = line

                        if smiles:
                            yield smiles
                            molecules_yielded += 1

                            if max_molecules and molecules_yielded >= max_molecules:
                                break

            if verbose:
                print(f"📊 Streamed {molecules_yielded:,} molecules from file")

        return file_generator()

    elif os.path.isdir(input_path):
        # Folder with multiple files
        file_list = find_smiles_files(input_path, verbose)
        if not file_list:
            raise FileNotFoundError(f"No SMILES files found in folder: {input_path}")

        return stream_smiles_from_files(file_list, max_molecules, verbose)

    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


def process_smiles_chunk(smiles_chunk, radius, use_features):
    """
    Process a chunk of SMILES strings and return fingerprint counts.

    Parameters:
    -----------
    smiles_chunk : list
        List of SMILES strings to process
    radius : int
        Morgan fingerprint radius
    use_features : bool
        Whether to use atom features

    Returns:
    --------
    tuple : (fingerprint_counts, valid_count, invalid_count)
    """
    fingerprint_counts = defaultdict(int)
    valid_molecules = 0
    invalid_molecules = 0

    for smiles in smiles_chunk:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_molecules += 1
                continue

            # Generate Morgan fingerprints
            info = {}
            AllChem.GetMorganFingerprint(
                mol,
                radius=radius,
                bitInfo=info,
                includeRedundantEnvironments=True,
                useFeatures=use_features,
            )

            # Collect fingerprints for this radius level
            for fingerprint_hash, atom_info_list in info.items():
                for atom_idx, radius_level in atom_info_list:
                    if radius_level == radius:
                        fingerprint_counts[fingerprint_hash] += 1

            valid_molecules += 1

        except Exception:
            invalid_molecules += 1

    return fingerprint_counts, valid_molecules, invalid_molecules


def create_guacamol_vocabularies(
    smiles_input: str,
    output_dir: str = "vocabularies",
    radii: list = [0, 1, 2],
    use_features_options: list = [False, True],
    max_molecules: int = None,
    verbose: bool = True,
    n_processes: int = None,
    chunk_size: int = None,
    batch_size: int = None,
):
    """
    Create vocabulary files for GuacaMol dataset with different Morgan fingerprint settings.

    Parameters:
    -----------
    smiles_input : str
        Path to SMILES file or folder containing SMILES files
    output_dir : str
        Directory to save vocabulary files
    radii : list
        List of Morgan fingerprint radii to generate vocabularies for
    use_features_options : list
        List of useFeatures settings [False for structural, True for functional]
    max_molecules : int, optional
        Limit number of molecules to process (for testing)
    verbose : bool
        Print progress information
    n_processes : int, optional
        Number of processes to use (default: CPU count)
    chunk_size : int, optional
        Size of chunks for multiprocessing (default: auto-calculated)
    batch_size : int, optional
        Size of batches to process from stream (default: 100,000 for memory efficiency)

    Returns:
    --------
    dict : Dictionary with vocabulary statistics
    """

    # Set default number of processes
    if n_processes is None:
        n_processes = mp.cpu_count()

    # Set default batch size for memory efficiency
    if batch_size is None:
        batch_size = 100000  # Process 100K molecules at a time

    if verbose:
        print(f"🚀 Using {n_processes} processes for parallel processing")
        print(f"💾 Batch size: {batch_size:,} molecules (for memory efficiency)")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # # Get SMILES stream
    # smiles_stream = get_smiles_stream(smiles_input, max_molecules, verbose)

    # Calculate optimal chunk size if not provided
    if chunk_size is None:
        # For batch processing, use smaller chunks
        chunk_size = max(1000, batch_size // (n_processes * 4))

    if verbose:
        print(f"📦 Chunk size: {chunk_size:,} molecules per chunk")

    # Statistics storage
    vocab_stats = {}

    # Generate vocabularies for each combination of radius and useFeatures
    for radius in radii:
        for use_features in use_features_options:

            env_type = "functional" if use_features else "structural"
            vocab_name = f"vocabulary_radius{radius}_{env_type}_guacamol_v1"

            if verbose:
                print(f"\n🔬 Creating {vocab_name}")
                print(f"   Radius: {radius}, UseFeatures: {use_features} ({env_type})")

            # Process in batches to manage memory
            fingerprint_counts = defaultdict(int)
            valid_molecules = 0
            invalid_molecules = 0
            batch_count = 0

            # Restart stream for each vocabulary
            smiles_stream = get_smiles_stream(
                smiles_input, max_molecules, verbose=False
            )

            current_batch = []
            for smiles in smiles_stream:
                current_batch.append(smiles)

                # Process when batch is full
                if len(current_batch) >= batch_size:
                    batch_count += 1
                    if verbose:
                        print(
                            f"   📊 Processing batch {batch_count} ({len(current_batch):,} molecules)..."
                        )

                    # Create chunks from current batch
                    batch_chunks = [
                        current_batch[i : i + chunk_size]
                        for i in range(0, len(current_batch), chunk_size)
                    ]

                    # Process chunks in parallel
                    process_func = partial(
                        process_smiles_chunk, radius=radius, use_features=use_features
                    )

                    with mp.Pool(n_processes) as pool:
                        if verbose:
                            results = list(
                                tqdm(
                                    pool.imap(process_func, batch_chunks),
                                    total=len(batch_chunks),
                                    desc=f"  Batch {batch_count}",
                                )
                            )
                        else:
                            results = pool.map(process_func, batch_chunks)

                    # Combine results from batch
                    for chunk_fingerprints, chunk_valid, chunk_invalid in results:
                        for fingerprint_hash, count in chunk_fingerprints.items():
                            fingerprint_counts[fingerprint_hash] += count

                        valid_molecules += chunk_valid
                        invalid_molecules += chunk_invalid

                    # Clear batch to free memory
                    current_batch = []

            # Process remaining molecules in final batch
            if current_batch:
                batch_count += 1
                if verbose:
                    print(
                        f"   📊 Processing final batch {batch_count} ({len(current_batch):,} molecules)..."
                    )

                batch_chunks = [
                    current_batch[i : i + chunk_size]
                    for i in range(0, len(current_batch), chunk_size)
                ]

                process_func = partial(
                    process_smiles_chunk, radius=radius, use_features=use_features
                )

                with mp.Pool(n_processes) as pool:
                    if verbose:
                        results = list(
                            tqdm(
                                pool.imap(process_func, batch_chunks),
                                total=len(batch_chunks),
                                desc=f"  Final batch",
                            )
                        )
                    else:
                        results = pool.map(process_func, batch_chunks)

                for chunk_fingerprints, chunk_valid, chunk_invalid in results:
                    for fingerprint_hash, count in chunk_fingerprints.items():
                        fingerprint_counts[fingerprint_hash] += count

                    valid_molecules += chunk_valid
                    invalid_molecules += chunk_invalid

            if verbose:
                print(f"   ✓ Valid molecules: {valid_molecules:,}")
                print(f"   ✗ Invalid molecules: {invalid_molecules:,}")
                print(
                    f"   🔍 Unique atom environments found: {len(fingerprint_counts):,}"
                )

            # Create vocabulary dictionary
            # Sort by frequency (most common first) for better token assignments
            sorted_fingerprints = sorted(
                fingerprint_counts.items(), key=lambda x: x[1], reverse=True
            )

            vocab_dict = {}
            # Start from 1 to reserve 0 for PAD token
            for token_id, (fingerprint_hash, count) in enumerate(
                sorted_fingerprints, start=1
            ):
                vocab_dict[fingerprint_hash] = token_id

            # Add special tokens
            vocab_dict["PAD"] = 0
            vocab_dict["MASK"] = max(vocab_dict.values()) + 1
            vocab_dict["UNK"] = max(vocab_dict.values()) + 1
            vocab_dict["CLS"] = max(vocab_dict.values()) + 1

            # Save vocabulary
            vocab_filename = f"{vocab_name}.pkl"
            vocab_path = os.path.join(output_dir, vocab_filename)

            with open(vocab_path, "wb") as f:
                pickle.dump(vocab_dict, f)
            # Save vocabulary
            vocab_filename_count = f"{vocab_name}_count.pkl"
            vocab_path_count = os.path.join(output_dir, vocab_filename_count)

            with open(vocab_path_count, "wb") as f:
                pickle.dump(sorted_fingerprints, f)
            # Store statistics
            vocab_stats[vocab_name] = {
                "radius": radius,
                "use_features": use_features,
                "environment_type": env_type,
                "total_fingerprints": len(sorted_fingerprints),
                "total_tokens": len(vocab_dict),
                "valid_molecules": valid_molecules,
                "invalid_molecules": invalid_molecules,
                "most_common_fingerprints": sorted_fingerprints[:10],  # Top 10
                "file_path": vocab_path,
                "file_size_mb": os.path.getsize(vocab_path) / (1024 * 1024),
            }

            if verbose:
                print(
                    f"   ✅ Created vocabulary with {len(sorted_fingerprints):,} atom environments"
                )
                print(f"   📁 Saved to: {vocab_path}")
                print(
                    f"   💾 File size: {vocab_stats[vocab_name]['file_size_mb']:.2f} MB"
                )

    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("📈 VOCABULARY CREATION SUMMARY")
        print("=" * 60)

        for vocab_name, stats in vocab_stats.items():
            print(f"\n🔍 {vocab_name}:")
            print(f"   • Environment type: {stats['environment_type']}")
            print(f"   • Radius: {stats['radius']}")
            print(f"   • Total atom environments: {stats['total_fingerprints']:,}")
            print(f"   • Total tokens (with special): {stats['total_tokens']:,}")
            print(f"   • File size: {stats['file_size_mb']:.2f} MB")

    return vocab_stats


def analyze_vocabulary_differences(vocab_stats: dict, verbose: bool = True):
    """
    Analyze differences between structural and functional vocabularies.
    """
    if not verbose:
        return

    print("\n" + "=" * 60)
    print("🔬 STRUCTURAL vs FUNCTIONAL COMPARISON")
    print("=" * 60)

    for radius in [0, 1, 2]:
        struct_key = f"vocabulary_radius{radius}_structural_guacamol_v1"
        func_key = f"vocabulary_radius{radius}_functional_guacamol_v1"

        if struct_key in vocab_stats and func_key in vocab_stats:
            struct_count = vocab_stats[struct_key]["total_fingerprints"]
            func_count = vocab_stats[func_key]["total_fingerprints"]

            print(f"\n📏 Radius {radius}:")
            print(f"   • Structural environments: {struct_count:,}")
            print(f"   • Functional environments: {func_count:,}")
            print(f"   • Difference: {abs(struct_count - func_count):,}")
            print(f"   • Ratio (func/struct): {func_count/struct_count:.2f}")


def load_and_inspect_vocabulary(vocab_path: str):
    """
    Load and inspect a vocabulary file.
    """
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Separate special tokens from fingerprint tokens
    special_tokens = {k: v for k, v in vocab.items() if isinstance(k, str)}
    fingerprint_tokens = {k: v for k, v in vocab.items() if isinstance(k, int)}

    print(f"📚 Vocabulary: {vocab_path}")
    print(f"   • Total entries: {len(vocab):,}")
    print(f"   • Fingerprint tokens: {len(fingerprint_tokens):,}")
    print(f"   • Special tokens: {special_tokens}")
    print(f"   • Token ID range: {min(vocab.values())} to {max(vocab.values())}")

    return vocab


def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(description="Create GuacaMol vocabularies")
    parser.add_argument(
        "--smiles_input",
        type=str,
        default="data/guacamol_v1_all.smiles",
        help="Path to SMILES file or folder containing SMILES files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mole/data/vocabularies",
        help="Output directory for vocabulary files",
    )
    parser.add_argument(
        "--radii",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Morgan fingerprint radii to process",
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
        help="Skip functional environment vocabularies",
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
        help="Size of chunks for multiprocessing (default: auto-calculated)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Size of batches to process from stream (default: 100,000 for memory efficiency)",
    )

    args = parser.parse_args()

    # Set use_features options
    use_features_options = [False]
    if not args.no_functional:
        use_features_options.append(True)

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

    # Analyze differences
    analyze_vocabulary_differences(vocab_stats, verbose=True)


if __name__ == "__main__":
    main()
