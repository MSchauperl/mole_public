#!/usr/bin/env python3
"""
Test script to verify fold-based splits work correctly with the ChemBL datamodule.
"""

import sys
import warnings
from pathlib import Path

# Suppress RDKit deprecation warnings
warnings.filterwarnings("ignore", message=".*please use MorganGenerator.*")

# Add the mole package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mole.data.chembl_datamodule import ChemBLDataModule


def test_fold_based_splits():
    """Test that fold-based splits work correctly."""
    print("🧪 Testing fold-based splits with ChemBL datamodule...")

    # Check if filtered dataset exists
    filtered_dir = Path("data/ChemBl_filtered")
    if not filtered_dir.exists():
        print("❌ Error: Filtered dataset not found!")
        print("Please run create_filtered_chembl_dataset_with_folds.py first.")
        return False

    # Create datamodule with fold-based splits
    data_module = ChemBLDataModule(
        chembl_smiles_path="data/ChemBl_filtered/chemblSmiles_top10_min3.pckl",
        chembl_labels_path="data/ChemBl_filtered/labelsHard_top10_min3.pckl",
        chembl_target_names_path="data/ChemBl_filtered/targetNames_top10_min3.txt",
        chembl_compound_names_path="data/ChemBl_filtered/compoundNames_top10_min3.txt",
        split_indices_path="data/ChemBl_filtered/splits_top10_min3.pckl",
        input_vocab_path="mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        target_vocab_path="mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
        batch_size=4,  # Small batch for testing
        num_workers=0,  # No multiprocessing for testing
    )

    print("📊 Setting up datamodule...")
    data_module.setup("fit")

    # Check dataset sizes
    train_size = len(data_module.train_dataset)
    val_size = len(data_module.val_dataset)

    print(f"✅ Dataset setup successful!")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    print(f"  Total: {train_size + val_size}")

    # Test that we can get a batch
    print("\n🔍 Testing batch creation...")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Get one batch from each
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print(f"✅ Batch creation successful!")
    print(f"  Training batch keys: {list(train_batch.keys())}")
    print(f"  Validation batch keys: {list(val_batch.keys())}")

    # Check that batches have the expected structure for MOLE graph-based approach
    expected_keys = [
        "x",  # Node features
        "edge_index",  # Edge indices
        "batch",  # Batch assignment
        "labels",  # MLM labels
        "chembl_targets",  # ChemBL classification targets
        "chembl_mask",  # Mask for ChemBL targets
        "smiles",  # SMILES strings
    ]

    missing_keys = []
    for key in expected_keys:
        if key not in train_batch:
            missing_keys.append(f"train_{key}")
        if key not in val_batch:
            missing_keys.append(f"val_{key}")

    if missing_keys:
        print(f"❌ Missing keys: {missing_keys}")
        return False

    print(f"✅ All expected batch keys present!")

    # Check batch shapes
    print(f"\n📐 Batch shapes:")
    print(f"  Training x (node features): {train_batch['x'].shape}")
    print(f"  Training labels: {train_batch['labels'].shape}")
    print(f"  Training chembl_targets: {train_batch['chembl_targets'].shape}")
    print(f"  Validation x (node features): {val_batch['x'].shape}")
    print(f"  Validation labels: {val_batch['labels'].shape}")
    print(f"  Validation chembl_targets: {val_batch['chembl_targets'].shape}")

    # Check that labels are binary (-1, 0, 1)
    train_labels = train_batch["labels"]
    val_labels = val_batch["labels"]

    unique_train_labels = set(train_labels.flatten().tolist())
    unique_val_labels = set(val_labels.flatten().tolist())

    print(f"  Training label values: {sorted(unique_train_labels)}")
    print(f"  Validation label values: {sorted(unique_val_labels)}")

    # Verify no overlap between train and validation
    print(f"\n🔒 Checking for data leakage...")

    # Get SMILES from both datasets
    train_smiles = data_module.train_dataset.smiles
    val_smiles = data_module.val_dataset.smiles

    train_smiles_set = set(train_smiles)
    val_smiles_set = set(val_smiles)

    overlap = train_smiles_set.intersection(val_smiles_set)

    if len(overlap) > 0:
        print(
            f"❌ Data leakage detected! {len(overlap)} SMILES appear in both train and validation"
        )
        print(f"  Example overlapping SMILES: {list(overlap)[:3]}")
        return False
    else:
        print(f"✅ No data leakage detected!")
        print(f"  Training SMILES: {len(train_smiles_set)} unique")
        print(f"  Validation SMILES: {len(val_smiles_set)} unique")
        print(f"  Overlap: {len(overlap)}")

    print(f"\n🎉 All tests passed! Fold-based splits are working correctly.")
    return True


def test_random_splits():
    """Test that random splits still work when split_indices_path is None."""
    print("\n🧪 Testing random splits (fallback mode)...")

    # Create datamodule without split indices (should use random splits)
    data_module = ChemBLDataModule(
        chembl_smiles_path="data/ChemBl_filtered/chemblSmiles_top10_min3.pckl",
        chembl_labels_path="data/ChemBl_filtered/labelsHard_top10_min3.pckl",
        chembl_target_names_path="data/ChemBl_filtered/targetNames_top10_min3.txt",
        chembl_compound_names_path="data/ChemBl_filtered/compoundNames_top10_min3.txt",
        # No split_indices_path - should use random splits
        input_vocab_path="mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        target_vocab_path="mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
        validation_split=0.2,  # 20% validation
        test_split=0.1,  # 10% test
        batch_size=4,
        num_workers=0,
    )

    print("📊 Setting up datamodule with random splits...")
    data_module.setup("fit")

    train_size = len(data_module.train_dataset)
    val_size = len(data_module.val_dataset)

    print(f"✅ Random splits setup successful!")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    print(f"  Total: {train_size + val_size}")

    return True


def main():
    """Main test function."""
    print("🚀 Testing ChemBL datamodule with fold-based splits")
    print("=" * 60)

    # Test fold-based splits
    fold_success = test_fold_based_splits()

    # Test random splits
    random_success = test_random_splits()

    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"  Fold-based splits: {'✅ PASS' if fold_success else '❌ FAIL'}")
    print(f"  Random splits: {'✅ PASS' if random_success else '❌ FAIL'}")

    if fold_success and random_success:
        print("\n🎉 All tests passed! The datamodule is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
