#!/usr/bin/env python3
"""
Cleanup Script for Old ChemBL Training Scripts

This script helps remove old ChemBL training scripts after migrating to the unified approach.
It provides a safe way to backup and remove the old scripts.

Usage:
    python cleanup_old_scripts.py --dry-run  # Show what would be removed
    python cleanup_old_scripts.py --backup   # Backup and remove old scripts
    python cleanup_old_scripts.py --remove   # Remove old scripts directly
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List


# Old scripts that can be replaced by run_chembl_unified.py
OLD_SCRIPTS = [
    "run_chembl_only_t4.py",
    "run_chembl_training_t4.py", 
    "run_chembl_filtered_only.py",
    "run_chembl_filtered_training.py",
    "run_chembl_training_test.py",
    "run_chembl_t4_folds.py",
    "run_chembl_finetune_t4.py",
]

# Scripts to keep
KEEP_SCRIPTS = [
    "run_chembl_unified.py",
    "run_chembl_filtered_fold_training.py",  # Advanced fold-based training
    "chembl_config_base.py",
    "migrate_to_unified.py",
    "cleanup_old_scripts.py",
    "README_CHEMBL_SCRIPTS.md",
]


def get_script_paths() -> List[Path]:
    """Get paths of old scripts that exist in the current directory."""
    current_dir = Path.cwd()
    existing_scripts = []
    
    for script in OLD_SCRIPTS:
        script_path = current_dir / script
        if script_path.exists():
            existing_scripts.append(script_path)
    
    return existing_scripts


def dry_run():
    """Show what would be removed without actually removing anything."""
    existing_scripts = get_script_paths()
    
    print("🔍 Dry Run - Scripts that would be removed:")
    print("=" * 50)
    
    if not existing_scripts:
        print("✅ No old scripts found to remove!")
        return
    
    for script_path in existing_scripts:
        print(f"❌ {script_path.name}")
    
    print(f"\n📊 Summary: {len(existing_scripts)} script(s) would be removed")
    print("\n💡 These scripts can be replaced by:")
    print("   python run_chembl_unified.py --dataset <dataset> --mlm/--no-mlm --splits <splits>")
    print("\n📖 Run 'python migrate_to_unified.py --show-all' for detailed migration guide")


def backup_and_remove():
    """Backup old scripts to a backup directory and then remove them."""
    existing_scripts = get_script_paths()
    
    if not existing_scripts:
        print("✅ No old scripts found to remove!")
        return
    
    # Create backup directory
    backup_dir = Path.cwd() / "backup_old_scripts"
    backup_dir.mkdir(exist_ok=True)
    
    print(f"📦 Backing up {len(existing_scripts)} script(s) to {backup_dir}")
    print("=" * 50)
    
    for script_path in existing_scripts:
        backup_path = backup_dir / script_path.name
        shutil.copy2(script_path, backup_path)
        print(f"📋 Backed up: {script_path.name}")
    
    print(f"\n🗑️  Removing old scripts...")
    for script_path in existing_scripts:
        script_path.unlink()
        print(f"❌ Removed: {script_path.name}")
    
    print(f"\n✅ Cleanup complete!")
    print(f"📁 Backups saved to: {backup_dir}")
    print(f"💡 You can restore scripts from backup if needed")


def remove_directly():
    """Remove old scripts directly without backup."""
    existing_scripts = get_script_paths()
    
    if not existing_scripts:
        print("✅ No old scripts found to remove!")
        return
    
    print(f"🗑️  Removing {len(existing_scripts)} script(s) directly:")
    print("=" * 50)
    
    for script_path in existing_scripts:
        script_path.unlink()
        print(f"❌ Removed: {script_path.name}")
    
    print(f"\n✅ Cleanup complete!")
    print("⚠️  Note: Scripts were removed without backup")


def show_status():
    """Show current status of scripts in the directory."""
    current_dir = Path.cwd()
    
    print("📊 Current Script Status:")
    print("=" * 50)
    
    # Check old scripts
    print("🔴 Old scripts (can be removed):")
    old_found = False
    for script in OLD_SCRIPTS:
        script_path = current_dir / script
        if script_path.exists():
            print(f"  ❌ {script}")
            old_found = True
        else:
            print(f"  ✅ {script} (not found)")
    
    if not old_found:
        print("  ✅ All old scripts already removed!")
    
    print("\n🟢 Scripts to keep:")
    for script in KEEP_SCRIPTS:
        script_path = current_dir / script
        if script_path.exists():
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script} (missing)")
    
    print("\n💡 Recommended action:")
    if old_found:
        print("  Run: python cleanup_old_scripts.py --backup")
    else:
        print("  No cleanup needed - all old scripts already removed!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Cleanup old ChemBL training scripts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing anything",
    )
    
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup old scripts and then remove them (recommended)",
    )
    
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove old scripts directly without backup",
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status of scripts in the directory",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    action_count = sum([args.dry_run, args.backup, args.remove, args.status])
    if action_count == 0:
        parser.print_help()
        print("\n💡 Quick start:")
        print("python cleanup_old_scripts.py --status")
        return
    elif action_count > 1:
        print("❌ Error: Please specify only one action")
        return
    
    # Execute requested action
    if args.dry_run:
        dry_run()
    elif args.backup:
        backup_and_remove()
    elif args.remove:
        remove_directly()
    elif args.status:
        show_status()


if __name__ == "__main__":
    main()
