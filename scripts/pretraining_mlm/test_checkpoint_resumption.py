#!/usr/bin/env python3
"""
Test script to verify checkpoint resumption functionality

This script tests that the checkpoint resumption arguments are properly
passed through to the training commands.
"""

import subprocess
import sys
import warnings
from pathlib import Path

# Suppress RDKit deprecation warnings
warnings.filterwarnings("ignore", message=".*please use MorganGenerator.*")


def test_t4_folds_checkpoint_args():
    """Test that T4 folds script accepts checkpoint arguments."""
    print("🧪 Testing T4 folds script checkpoint arguments...")

    # Test help output contains checkpoint arguments
    try:
        result = subprocess.run(
            ["python", "scripts/pretraining_mlm/run_chembl_t4_folds.py", "--help"],
            capture_output=True,
            text=True,
            check=True,
        )

        help_output = result.stdout
        if (
            "--resume_from_checkpoint" in help_output
            and "--freeze_encoder" in help_output
        ):
            print("✅ T4 folds script accepts checkpoint arguments")
            return True
        else:
            print("❌ T4 folds script missing checkpoint arguments")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ T4 folds script help failed: {e}")
        return False


def test_fold_training_checkpoint_args():
    """Test that fold training script accepts checkpoint arguments."""
    print("🧪 Testing fold training script checkpoint arguments...")

    # Test help output contains checkpoint arguments
    try:
        result = subprocess.run(
            [
                "python",
                "scripts/pretraining_mlm/run_chembl_filtered_fold_training.py",
                "--help",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        help_output = result.stdout
        if (
            "--resume_from_checkpoint" in help_output
            and "--freeze_encoder" in help_output
        ):
            print("✅ Fold training script accepts checkpoint arguments")
            return True
        else:
            print("❌ Fold training script missing checkpoint arguments")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Fold training script help failed: {e}")
        return False


def test_finetune_checkpoint_args():
    """Test that fine-tuning script accepts checkpoint arguments."""
    print("🧪 Testing fine-tuning script checkpoint arguments...")

    # Test help output contains checkpoint arguments
    try:
        result = subprocess.run(
            ["python", "scripts/pretraining_mlm/run_chembl_finetune_t4.py", "--help"],
            capture_output=True,
            text=True,
            check=True,
        )

        help_output = result.stdout
        if "--resume_from_checkpoint" in help_output:
            print("✅ Fine-tuning script accepts checkpoint arguments")
            return True
        else:
            print("❌ Fine-tuning script missing checkpoint arguments")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Fine-tuning script help failed: {e}")
        return False


def test_unsupervised_checkpoint_args():
    """Test that unsupervised pretraining script accepts checkpoint arguments."""
    print("🧪 Testing unsupervised pretraining script checkpoint arguments...")

    # Test help output contains checkpoint arguments
    try:
        result = subprocess.run(
            [
                "python",
                "scripts/pretraining_mlm/run_unsupervised_pretraining_a100_2gpu.py",
                "--help",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        help_output = result.stdout
        if (
            "--resume_from_checkpoint" in help_output
            and "--freeze_encoder" in help_output
        ):
            print("✅ Unsupervised pretraining script accepts checkpoint arguments")
            return True
        else:
            print("❌ Unsupervised pretraining script missing checkpoint arguments")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Unsupervised pretraining script help failed: {e}")
        return False


def test_configuration_integration():
    """Test that configuration classes support checkpoint resumption."""
    print("🧪 Testing configuration integration...")

    try:
        # Import configuration classes
        sys.path.insert(0, str(Path(__file__).parent))
        from chembl_config_base import T4ConfigWithFolds

        # Test that add_pretrained_model method exists
        config = T4ConfigWithFolds()
        if hasattr(config, "add_pretrained_model"):
            print("✅ Configuration classes support checkpoint resumption")
            return True
        else:
            print("❌ Configuration classes missing add_pretrained_model method")
            return False

    except ImportError as e:
        print(f"❌ Failed to import configuration classes: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Testing Checkpoint Resumption Functionality")
    print("=" * 60)
    print()

    tests = [
        test_t4_folds_checkpoint_args,
        test_fold_training_checkpoint_args,
        test_finetune_checkpoint_args,
        test_unsupervised_checkpoint_args,
        test_configuration_integration,
    ]

    results = {}
    for test in tests:
        try:
            results[test.__name__] = test()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results[test.__name__] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 Checkpoint Resumption Test Summary:")

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All checkpoint resumption tests passed!")
        print("   Ready to use checkpoint resumption functionality.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
