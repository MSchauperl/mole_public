#!/usr/bin/env python3
"""
Shared base configuration for ChemBL training scripts.

This module provides common configuration parameters and utilities
to reduce code redundancy across different ChemBL training scripts.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


class ChemBLConfig:
    """Base configuration class for ChemBL training scripts."""
    
    def __init__(self, config_name: str = "base"):
        self.config_name = config_name
        self.base_params = self._get_base_params()
        
    def _get_base_params(self) -> Dict[str, str]:
        """Get base parameters common to all ChemBL training configurations."""
        return {
            # Data paths
            "--chembl_smiles_path": "data/ChemBl/chembl20Smiles.pckl",
            "--chembl_labels_path": "data/ChemBl/labelsHard.pckl", 
            "--chembl_target_names_path": "data/ChemBl/labelsWeakHard.targetNames",
            "--chembl_compound_names_path": "data/ChemBl/labelsWeakHard.cmpNames",
            "--input_vocab": "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
            "--target_vocab": "mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
            
            # Environment configuration
            "--input_radius": "0",
            "--target_radius": "1",
            "--target_use_features": "",  # Flag for functional environments
            
            # Common training settings
            "--validation_split": "0.1",
            "--test_split": "0.1",
            "--dropout": "0.1",
            "--classifier_dropout": "0.1",
            "--weight_decay": "0.01",
            "--gradient_clip_val": "1.0",
            "--seed": "42",
            "--log_predictions": "",
            "--log_target_metrics": "",
        }
    
    def get_config(self, overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get the full configuration with optional overrides."""
        config = self.base_params.copy()
        if overrides:
            config.update(overrides)
        return config
    
    def run_training(self, config: Dict[str, str], script_name: str = "train_chembl_mlm.py"):
        """Run the training script with the given configuration."""
        # Build command
        script_path = Path(__file__).parent.parent.parent / "mole" / "cli" / script_name
        cmd = [sys.executable, str(script_path)]
        
        # Add parameters
        for key, value in config.items():
            cmd.append(key)
            if value:  # Only add value if it's not an empty string (flag)
                cmd.append(value)
        
        # Add any additional arguments passed to the script
        cmd.extend(sys.argv[1:])
        
        return cmd
    
    def print_config_summary(self, config: Dict[str, str], title: str):
        """Print a formatted summary of the configuration."""
        print(f"🚀 {title}")
        print("=" * 70)
        
        # Dataset info
        max_samples = config.get("--max_samples", "456331")
        if max_samples != "456331":
            print(f"📊 Dataset: ChemBL (LIMITED TO {max_samples} molecules)")
        else:
            print("📊 Dataset: ChemBL (~456K molecules)")
        
        # GPU info
        gpu_info = self._get_gpu_info(config)
        print(f"🎯 {gpu_info}")
        
        # Architecture info
        hidden_size = config["--hidden_size"]
        num_layers = config["--num_hidden_layers"]
        arch_desc = self._get_architecture_description(config)
        print(f"🏗️  Architecture: {hidden_size} hidden, {num_layers} layers ({arch_desc})")
        
        # Task info
        target_radius = config["--target_radius"]
        input_radius = config["--input_radius"]
        max_targets = config.get("--max_targets", "N/A")
        min_activity = config.get("--min_target_activity", "N/A")
        
        # Check if ChemBL-only mode
        if "--chembl_only" in config:
            print(f"🎯 Task: ChemBL classification only (radius-{input_radius} structural input)")
        else:
            print(f"🎯 Task: Predict radius-{target_radius} functional from radius-{input_radius} structural")
        
        if max_targets != "N/A":
            print(f"🧪 ChemBL: Max {max_targets} targets, min {min_activity} activities per target")
        
        # Training info
        batch_size = config["--batch_size"]
        grad_batches = config["--accumulate_grad_batches"]
        effective_batch = int(batch_size) * int(grad_batches)
        print(f"💾 Batch size: {batch_size} × {grad_batches} = {effective_batch} effective")
        print(f"⚡ Precision: {config['--precision']}-bit")
        print(f"🧠 Max sequence length: {config['--max_length']}")
        print(f"👥 Workers: {config['--num_workers']}")
        
        # Loss info
        self._print_loss_info(config)
        
        print(f"📁 Output: {config['--output_dir']}")
        
        # Special warnings
        self._print_special_warnings(config)
        
        print("=" * 70)
        print()
    
    def _get_gpu_info(self, config: Dict[str, str]) -> str:
        """Get GPU information string based on configuration."""
        if "t4" in config.get("--model_name", "").lower():
            return "GPU: Tesla T4 (16 GB VRAM) - Memory Optimized Configuration"
        elif "a100" in config.get("--model_name", "").lower():
            return "GPU: NVIDIA A100 (40 GB VRAM) - High Performance Configuration"
        else:
            return "GPU: Generic Configuration"
    
    def _get_architecture_description(self, config: Dict[str, str]) -> str:
        """Get architecture description based on configuration."""
        hidden_size = int(config["--hidden_size"])
        if hidden_size <= 128:
            return "Tiny for Speed"
        elif hidden_size <= 512:
            return "T4 Memory Optimized"
        elif hidden_size >= 768:
            return "Standard Size"
        else:
            return "Custom Size"
    
    def _print_loss_info(self, config: Dict[str, str]):
        """Print loss weighting information."""
        if "--chembl_only" in config:
            cls_weight = config.get("--classification_loss_weight", "1.0")
            print(f"⚖️  Loss: Classification only (weight={cls_weight})")
            print("🚫 MLM: DISABLED (ChemBL-only mode)")
        else:
            mlm_weight = config.get("--mlm_loss_weight", "1.0")
            cls_weight = config.get("--classification_loss_weight", "1.0")
            print(f"⚖️  Loss weights: MLM={mlm_weight}, Classification={cls_weight}")
    
    def _print_special_warnings(self, config: Dict[str, str]):
        """Print any special warnings based on configuration."""
        if config.get("--max_samples") == "1000":
            print("⚠️  WARNING: This is for rapid testing only - tiny model & dataset!")
        elif int(config.get("--max_samples", "999999")) < 50000:
            print("⚠️  NOTE: Limited dataset size for testing purposes")


class T4Config(ChemBLConfig):
    """Tesla T4 optimized configuration."""
    
    def __init__(self):
        super().__init__("t4")
    
    def get_config(self, overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get T4-optimized configuration."""
        t4_params = {
            "--output_dir": "outputs/chembl_mlm",
            "--model_name": "chembl_r0_to_r1_functional_t4_optimized",
            "--max_targets": "50",
            "--min_target_activity": "50",
            "--hidden_size": "512",
            "--num_hidden_layers": "8",
            "--num_attention_heads": "8",
            "--intermediate_size": "2048",
            "--batch_size": "4",
            "--learning_rate": "1e-4",
            "--warmup_steps": "3000",
            "--max_epochs": "20",
            "--val_check_interval": "0.25",
            "--patience": "3",
            "--mlm_loss_weight": "0.1",
            "--classification_loss_weight": "1.0",
            "--gpus": "1",
            "--num_workers": "4",
            "--precision": "16",
            "--accumulate_grad_batches": "16",
            "--max_length": "256",
            "--max_targets_to_log": "20",
            "--max_samples": "45633",  # 10% for testing
        }
        
        config = super().get_config(t4_params)
        if overrides:
            config.update(overrides)
        return config


class T4OnlyConfig(ChemBLConfig):
    """Tesla T4 ChemBL-only (no MLM) configuration."""
    
    def __init__(self):
        super().__init__("t4_only")
    
    def get_config(self, overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get T4 ChemBL-only configuration."""
        t4_only_params = {
            "--output_dir": "outputs/chembl_only",
            "--model_name": "chembl_only_t4_optimized",
            "--max_targets": "50",
            "--min_target_activity": "50",
            "--hidden_size": "768",  # Can be larger without MLM
            "--num_hidden_layers": "8",
            "--num_attention_heads": "12",
            "--intermediate_size": "2048",
            "--batch_size": "8",  # Can be larger without MLM
            "--learning_rate": "2e-4",
            "--warmup_steps": "1000",
            "--max_epochs": "20",
            "--val_check_interval": "0.25",
            "--patience": "5",
            "--chembl_only": "",  # Enable ChemBL-only mode
            "--classification_loss_weight": "1.0",
            "--gpus": "1",
            "--num_workers": "4",
            "--precision": "16",
            "--accumulate_grad_batches": "8",
            "--max_length": "256",
            "--max_targets_to_log": "20",
            "--max_samples": "45633",  # 10% for testing
        }
        
        config = super().get_config(t4_only_params)
        if overrides:
            config.update(overrides)
        return config


class TestConfig(ChemBLConfig):
    """Minimal test configuration for rapid testing."""
    
    def __init__(self):
        super().__init__("test")
    
    def get_config(self, overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get minimal test configuration."""
        test_params = {
            "--output_dir": "outputs/chembl_test",
            "--model_name": "chembl_test_tiny",
            "--max_targets": "10",
            "--min_target_activity": "50",
            "--hidden_size": "128",
            "--num_hidden_layers": "2",
            "--num_attention_heads": "2",
            "--intermediate_size": "256",
            "--batch_size": "4",
            "--learning_rate": "1e-3",
            "--warmup_steps": "50",
            "--max_epochs": "2",
            "--val_check_interval": "1.0",
            "--patience": "10",
            "--mlm_loss_weight": "1.0",
            "--classification_loss_weight": "1.0",
            "--gpus": "1",
            "--num_workers": "0",
            "--precision": "32",
            "--accumulate_grad_batches": "4",
            "--max_length": "64",
            "--max_targets_to_log": "5",
            "--max_samples": "1000",  # Only 1000 samples for rapid testing
        }
        
        config = super().get_config(test_params)
        if overrides:
            config.update(overrides)
        return config


def run_chembl_training(config_class, title: str, overrides: Optional[Dict[str, str]] = None):
    """Common function to run ChemBL training with any configuration."""
    config_instance = config_class()
    config = config_instance.get_config(overrides)
    
    # Print configuration summary
    config_instance.print_config_summary(config, title)
    
    # Build and display command
    cmd = config_instance.run_training(config)
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("⚠️  Training interrupted by user")
        sys.exit(1) 