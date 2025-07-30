#!/usr/bin/env python3
"""
Check GuacaMol Dataset Information

This script provides information about the GuacaMol dataset and
estimates training time and resource requirements.
"""

import os
import pickle


def count_smiles_lines(file_path):
    """Count the number of SMILES in the dataset file"""
    if not os.path.exists(file_path):
        return 0, f"File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            count = sum(1 for _ in f)
        return count, None
    except Exception as e:
        return 0, f"Error reading file: {e}"


def get_file_size_mb(file_path):
    """Get file size in MB"""
    if not os.path.exists(file_path):
        return 0
    return os.path.getsize(file_path) / (1024 * 1024)


def load_vocab_info(vocab_path):
    """Load vocabulary and get basic statistics"""
    if not os.path.exists(vocab_path):
        return None, f"Vocabulary not found: {vocab_path}"

    try:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        # Get special tokens
        special_tokens = {}
        for token in ["PAD", "MASK", "UNK", "CLS"]:
            if token in vocab:
                special_tokens[token] = vocab[token]

        vocab_size = len(vocab)
        return {
            "size": vocab_size,
            "special_tokens": special_tokens,
            "atom_envs": vocab_size - len(special_tokens),
        }, None
    except Exception as e:
        return None, f"Error loading vocabulary: {e}"


def estimate_training_time(
    num_molecules, batch_size=64, accumulate_batches=2, epochs=20
):
    """Estimate training time"""
    effective_batch_size = batch_size * accumulate_batches
    steps_per_epoch = num_molecules // effective_batch_size
    total_steps = steps_per_epoch * epochs

    # Rough estimates (will vary by hardware)
    seconds_per_step_gpu = 0.5  # Rough estimate for molecular data
    hours_estimated = (total_steps * seconds_per_step_gpu) / 3600

    return {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "estimated_hours": hours_estimated,
        "effective_batch_size": effective_batch_size,
    }


def main():
    print("🧬 GuacaMol Dataset Information")
    print("=" * 50)

    # Check dataset file
    dataset_path = "data/guacamol_v1_all.smiles"
    num_molecules, error = count_smiles_lines(dataset_path)

    if error:
        print(f"❌ {error}")
        return

    file_size = get_file_size_mb(dataset_path)
    print(f"📁 Dataset file: {dataset_path}")
    print(f"📊 Number of molecules: {num_molecules:,}")
    print(f"💾 File size: {file_size:.1f} MB")

    # Check vocabularies
    print("\n🔤 Vocabulary Information")
    print("-" * 30)

    vocabs = [
        (
            "Input (structural, radius 0)",
            "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        ),
        (
            "Target (functional, radius 1)",
            "mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
        ),
    ]

    for desc, vocab_path in vocabs:
        vocab_info, error = load_vocab_info(vocab_path)
        if error:
            print(f"❌ {desc}: {error}")
        else:
            file_size = get_file_size_mb(vocab_path)
            print(f"✅ {desc}:")
            print(f"   • Vocabulary size: {vocab_info['size']:,}")
            print(f"   • Atom environments: {vocab_info['atom_envs']:,}")
            print(f"   • Special tokens: {vocab_info['special_tokens']}")
            print(f"   • File size: {file_size:.1f} MB")

    # Training estimates
    print("\n⏱️  Training Estimates")
    print("-" * 30)

    estimates = estimate_training_time(num_molecules)
    print(f"📈 Steps per epoch: {estimates['steps_per_epoch']:,}")
    print(f"🎯 Total training steps: {estimates['total_steps']:,}")
    print(f"⏰ Estimated time: {estimates['estimated_hours']:.1f} hours")
    print(f"💾 Effective batch size: {estimates['effective_batch_size']}")

    # Memory estimates
    print("\n💾 Memory Estimates")
    print("-" * 30)
    print("🔍 Model size: ~768 hidden × 12 layers ≈ 110M parameters")
    print("🔍 Estimated model memory: ~2-3 GB (with mixed precision)")
    print("🔍 Recommended GPU memory: ≥8 GB")
    print("🔍 Batch size 64 × 2 accumulation should fit on most modern GPUs")

    # Recommendations
    print("\n💡 Recommendations")
    print("-" * 30)
    print("🚀 Use mixed precision (--precision 16) for faster training")
    print("⚡ Increase --num_workers if you have many CPU cores")
    print("📊 Monitor GPU utilization and adjust batch size if needed")
    print("💾 Consider checkpointing every 2 epochs for the large dataset")
    print("🔍 Validation every 0.25 epochs provides good monitoring")


if __name__ == "__main__":
    main()
