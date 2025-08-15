#!/usr/bin/env python3
"""
Migration Helper for ChemBL Training Scripts

This script helps convert old script commands to the new unified script format.
It can also show the equivalent command for any old script.

Usage:
    python migrate_to_unified.py --old-script run_chembl_training_t4.py
    python migrate_to_unified.py --show-all
    python migrate_to_unified.py --interactive
"""

import argparse
import sys
from typing import Dict, List, Tuple


# Mapping from old scripts to new unified commands
SCRIPT_MAPPINGS = {
    "run_chembl_only_t4.py": {
        "command": "python run_chembl_unified.py --dataset full --mlm false --splits random",
        "description": "ChemBL-only training on full dataset (no MLM)"
    },
    "run_chembl_training_t4.py": {
        "command": "python run_chembl_unified.py --dataset full --mlm true --splits random",
        "description": "MLM + classification training on full dataset"
    },
    "run_chembl_filtered_only.py": {
        "command": "python run_chembl_unified.py --dataset filtered --mlm false --splits folds",
        "description": "ChemBL-only training on filtered dataset (no MLM)"
    },
    "run_chembl_filtered_training.py": {
        "command": "python run_chembl_unified.py --dataset filtered --mlm true --splits folds",
        "description": "MLM + classification training on filtered dataset"
    },
    "run_chembl_training_test.py": {
        "command": "python run_chembl_unified.py --dataset test --mlm true --splits random",
        "description": "Test training with 1K samples"
    },
    "run_chembl_t4_folds.py": {
        "command": "python run_chembl_unified.py --dataset filtered --mlm true --splits folds",
        "description": "MLM + classification training with fold-based splits"
    },
    "run_chembl_finetune_t4.py": {
        "command": "python run_chembl_unified.py --resume_from_checkpoint <path> [--freeze_encoder]",
        "description": "Fine-tuning with checkpoint resumption"
    }
}

# Fine-tuning examples
FINETUNE_EXAMPLES = [
    {
        "old": "python run_chembl_finetune_t4.py --pretrained_path outputs/chembl_mlm/best_model.ckpt --freeze_encoder",
        "new": "python run_chembl_unified.py --dataset filtered --mlm true --splits folds --resume_from_checkpoint outputs/chembl_mlm/best_model.ckpt --freeze_encoder",
        "description": "Fine-tune with frozen encoder"
    },
    {
        "old": "python run_chembl_finetune_t4.py --pretrained_path outputs/chembl_mlm/best_model.ckpt --chembl_only",
        "new": "python run_chembl_unified.py --dataset filtered --mlm false --splits folds --resume_from_checkpoint outputs/chembl_mlm/best_model.ckpt --freeze_encoder",
        "description": "Fine-tune ChemBL-only model"
    },
    {
        "old": "python run_chembl_finetune_t4.py --pretrained_path outputs/chembl_mlm/best_model.ckpt --use_folds",
        "new": "python run_chembl_unified.py --dataset filtered --mlm true --splits folds --resume_from_checkpoint outputs/chembl_mlm/best_model.ckpt",
        "description": "Fine-tune with fold-based splits"
    }
]


def show_all_mappings():
    """Show all script mappings."""
    print("🔄 ChemBL Script Migration Guide")
    print("=" * 60)
    print()
    
    print("📋 Direct Script Replacements:")
    print("-" * 40)
    for old_script, info in SCRIPT_MAPPINGS.items():
        print(f"Old: {old_script}")
        print(f"New: {info['command']}")
        print(f"     {info['description']}")
        print()
    
    print("🔧 Fine-tuning Examples:")
    print("-" * 40)
    for example in FINETUNE_EXAMPLES:
        print(f"Old: {example['old']}")
        print(f"New: {example['new']}")
        print(f"     {example['description']}")
        print()
    
    print("💡 Quick Start Commands:")
    print("-" * 40)
    print("# Recommended starting point:")
    print("python run_chembl_unified.py --dataset filtered --mlm true --splits folds")
    print()
    print("# Classification only:")
    print("python run_chembl_unified.py --dataset filtered --mlm false --splits folds")
    print()
    print("# Quick test:")
    print("python run_chembl_unified.py --dataset test --mlm true --splits random")


def show_script_mapping(script_name: str):
    """Show the mapping for a specific script."""
    if script_name not in SCRIPT_MAPPINGS:
        print(f"❌ Error: Unknown script '{script_name}'")
        print("Available scripts:")
        for script in SCRIPT_MAPPINGS.keys():
            print(f"  - {script}")
        return
    
    info = SCRIPT_MAPPINGS[script_name]
    print(f"🔄 Migration for {script_name}")
    print("=" * 50)
    print(f"Old: python {script_name}")
    print(f"New: {info['command']}")
    print(f"Description: {info['description']}")
    
    # Show additional options if available
    if script_name == "run_chembl_finetune_t4.py":
        print("\n🔧 Fine-tuning Examples:")
        for example in FINETUNE_EXAMPLES:
            print(f"Old: {example['old']}")
            print(f"New: {example['new']}")
            print(f"     {example['description']}")


def interactive_mode():
    """Run in interactive mode to help users migrate."""
    print("🔄 Interactive ChemBL Script Migration Helper")
    print("=" * 50)
    print()
    
    while True:
        print("Available options:")
        print("1. Show migration for specific script")
        print("2. Show all migrations")
        print("3. Show quick start commands")
        print("4. Exit")
        print()
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\nAvailable scripts:")
            for i, script in enumerate(SCRIPT_MAPPINGS.keys(), 1):
                print(f"{i}. {script}")
            print()
            
            try:
                script_choice = int(input("Enter script number: ")) - 1
                script_name = list(SCRIPT_MAPPINGS.keys())[script_choice]
                print()
                show_script_mapping(script_name)
            except (ValueError, IndexError):
                print("❌ Invalid choice")
        elif choice == "2":
            print()
            show_all_mappings()
        elif choice == "3":
            print("\n💡 Quick Start Commands:")
            print("-" * 40)
            print("# Recommended starting point:")
            print("python run_chembl_unified.py --dataset filtered --mlm true --splits folds")
            print()
            print("# Classification only:")
            print("python run_chembl_unified.py --dataset filtered --mlm false --splits folds")
            print()
            print("# Quick test:")
            print("python run_chembl_unified.py --dataset test --mlm true --splits random")
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")
        
        print("\n" + "=" * 50 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Migration helper for ChemBL training scripts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--old-script",
        type=str,
        help="Show migration for specific old script",
    )
    
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all script migrations",
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.old_script:
        show_script_mapping(args.old_script)
    elif args.show_all:
        show_all_mappings()
    else:
        parser.print_help()
        print("\n💡 Quick start:")
        print("python migrate_to_unified.py --show-all")


if __name__ == "__main__":
    main()
