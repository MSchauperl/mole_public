import os
import pickle
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import argparse


def create_guacamol_vocabularies(
    smiles_file: str,
    output_dir: str = "vocabularies",
    radii: list = [0, 1, 2],
    use_features_options: list = [False, True],
    max_molecules: int = None,
    verbose: bool = True,
):
    """
    Create vocabulary files for GuacaMol dataset with different Morgan fingerprint settings.

    Parameters:
    -----------
    smiles_file : str
        Path to the SMILES file (e.g., 'data/guacamol_v1_all.smiles')
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

    Returns:
    --------
    dict : Dictionary with vocabulary statistics
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read SMILES file
    if verbose:
        print(f"📚 Reading SMILES from: {smiles_file}")

    with open(smiles_file, "r") as f:
        smiles_list = [line.strip() for line in f.readlines() if line.strip()]

    if max_molecules:
        smiles_list = smiles_list[:max_molecules]
        if verbose:
            print(f"🔢 Processing first {max_molecules} molecules")

    if verbose:
        print(f"📊 Total molecules to process: {len(smiles_list):,}")

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

            # Collect all unique atom environments
            fingerprint_counts = defaultdict(int)
            valid_molecules = 0
            invalid_molecules = 0

            # Process molecules with progress bar
            for smiles in tqdm(
                smiles_list, desc=f"Processing R{radius} {env_type[:4]}"
            ):
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

                except Exception as e:
                    invalid_molecules += 1
                    if verbose and invalid_molecules <= 5:  # Show first few errors
                        print(f"⚠️  Error processing SMILES '{smiles}': {e}")

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
                print(f"   ✓ Valid molecules: {valid_molecules:,}")
                print(f"   ✗ Invalid molecules: {invalid_molecules:,}")

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
        "--smiles_file",
        type=str,
        default="data/guacamol_v1_all.smiles",
        help="Path to SMILES file",
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

    args = parser.parse_args()

    # Set use_features options
    use_features_options = [False]
    if not args.no_functional:
        use_features_options.append(True)

    # Create vocabularies
    vocab_stats = create_guacamol_vocabularies(
        smiles_file=args.smiles_file,
        output_dir=args.output_dir,
        radii=args.radii,
        use_features_options=use_features_options,
        max_molecules=args.max_molecules,
        verbose=True,
    )

    # Analyze differences
    analyze_vocabulary_differences(vocab_stats, verbose=True)


if __name__ == "__main__":
    main()
