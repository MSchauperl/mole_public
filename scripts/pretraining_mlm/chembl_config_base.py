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
    
    def add_pretrained_model(self, pretrained_path: str, freeze_encoder: bool = False) -> Dict[str, str]:
        """
        Add pretrained model loading configuration.
        
        Args:
            pretrained_path: Path to the pretrained model checkpoint
            freeze_encoder: Whether to freeze the encoder layers during fine-tuning
            
        Returns:
            Dictionary with pretrained model configuration parameters
        """
        pretrained_config = {
            "--resume_from_checkpoint": pretrained_path,
        }
        
        if freeze_encoder:
            pretrained_config["--freeze_encoder"] = ""
            
        return pretrained_config
    
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
        
        # Pretrained model info
        self._print_pretrained_info(config)
        
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
    
    def _print_pretrained_info(self, config: Dict[str, str]):
        """Print pretrained model information."""
        if "--resume_from_checkpoint" in config and config["--resume_from_checkpoint"]:
            checkpoint_path = config["--resume_from_checkpoint"]
            print(f"🔄 Pretrained model: {checkpoint_path}")
            
            if "--freeze_encoder" in config:
                print("🧊 Encoder: FROZEN (fine-tuning mode)")
            else:
                print("🔥 Encoder: TRAINABLE (continued pretraining)")
    
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
            "--min_target_activity": "500",
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
            "--min_target_activity": "2000",
            "--hidden_size": "768",  # Can be larger without MLM
            "--num_hidden_layers": "8",
            "--num_attention_heads": "12",
            "--intermediate_size": "2048",
            "--batch_size": "8",  # Can be larger without MLM
            "--learning_rate": "2e-4",
            "--warmup_steps": "1000",
            "--max_epochs": "20",
            "--val_check_interval": "0.25",
            "--patience": "6",
            "--chembl_only": "",  # Enable ChemBL-only mode
            "--classification_loss_weight": "1.0",
            "--gpus": "1",
            "--num_workers": "4",
            "--precision": "16",
            "--accumulate_grad_batches": "32",
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
            "--min_target_activity": "500",
            "--hidden_size": "128",
            "--num_hidden_layers": "2",
            "--num_attention_heads": "2",
            "--intermediate_size": "256",
            "--batch_size": "4",
            "--learning_rate": "1e-3",
            "--warmup_steps": "50",
            "--max_epochs": "20",
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


class T4FineTuneConfig(T4Config):
    """Tesla T4 fine-tuning configuration with frozen encoder."""
    
    def __init__(self):
        super().__init__()
        self.config_name = "t4_finetune"
    
    def get_config(self, overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get T4 fine-tuning configuration with optimized settings."""
        # Start with base T4 config
        config = super().get_config()
        
        # Fine-tuning specific overrides
        finetune_params = {
            "--output_dir": "outputs/chembl_finetune",
            "--model_name": "chembl_finetune_t4",
            "--learning_rate": "5e-5",  # Lower learning rate for fine-tuning
            "--warmup_steps": "500",    # Shorter warmup for fine-tuning
            "--max_epochs": "10",       # Fewer epochs for fine-tuning
            "--patience": "3",          # More aggressive early stopping
            "--classification_loss_weight": "2.0",  # Higher weight on classification for fine-tuning
            "--mlm_loss_weight": "0.05",  # Lower weight on MLM for fine-tuning
        }
        
        config.update(finetune_params)
        if overrides:
            config.update(overrides)
        return config


class T4OnlyFineTuneConfig(T4OnlyConfig):
    """Tesla T4 ChemBL-only fine-tuning configuration."""
    
    def __init__(self):
        super().__init__()
        self.config_name = "t4_only_finetune"
    
    def get_config(self, overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get T4 ChemBL-only fine-tuning configuration."""
        # Start with base T4-only config
        config = super().get_config()
        
        # Fine-tuning specific overrides
        finetune_params = {
            "--output_dir": "outputs/chembl_only_finetune",
            "--model_name": "chembl_only_finetune_t4",
            "--learning_rate": "2e-5",  # Lower learning rate for fine-tuning
            "--warmup_steps": "200",    # Shorter warmup
            "--max_epochs": "8",        # Fewer epochs
            "--patience": "2",          # More aggressive early stopping
        }
        
        config.update(finetune_params)
        if overrides:
            config.update(overrides)
        return config


class FilteredChemBLConfig(ChemBLConfig):
    """Configuration for filtered ChemBL dataset (top 10 targets, min 3 measurements)."""
    
    def __init__(self):
        super().__init__("filtered_chembl")
    
    def _get_base_params(self) -> Dict[str, str]:
        """Get base parameters for filtered ChemBL dataset."""
        base_params = super()._get_base_params()
        
        # Override data paths to use filtered dataset
        base_params.update({
            "--chembl_smiles_path": "data/ChemBl_filtered/chemblSmiles_top10_min3.pckl",
            "--chembl_labels_path": "data/ChemBl_filtered/labelsHard_top10_min3.pckl", 
            "--chembl_target_names_path": "data/ChemBl_filtered/targetNames_top10_min3.txt",
            "--chembl_compound_names_path": "data/ChemBl_filtered/compoundNames_top10_min3.txt",
        })
        
        return base_params
    
    def get_config(self, overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get filtered ChemBL configuration."""
        filtered_params = {
            "--output_dir": "outputs/chembl_filtered",
            "--model_name": "chembl_filtered_top10_t4",
            "--max_targets": "10",  # All targets in filtered dataset
            "--min_target_activity": "0",  # No need to filter further
            "--hidden_size": "768",  # Can use larger model with fewer targets
            "--num_hidden_layers": "12",
            "--num_attention_heads": "12",
            "--intermediate_size": "3072",
            "--batch_size": "16",  # Can use larger batch with fewer targets
            "--learning_rate": "1e-4",
            "--warmup_steps": "2000",
            "--max_epochs": "50",  # Can train longer with smaller dataset
            "--val_check_interval": "0.2",
            "--patience": "8",
            "--mlm_loss_weight": "0.2",
            "--classification_loss_weight": "1.0",
            "--gpus": "1",
            "--num_workers": "8",
            "--precision": "16",
            "--accumulate_grad_batches": "4",
            "--max_length": "512",  # Can handle longer sequences
            "--max_targets_to_log": "10",  # Log all targets
            # Use full filtered dataset (no sampling)
        }
        
        config = super().get_config(filtered_params)
        if overrides:
            config.update(overrides)
        return config
    
    def print_config_summary(self, config: Dict[str, str], title: str):
        """Print a formatted summary with filtered dataset info."""
        print(f"🚀 {title}")
        print("=" * 70)
        
        # Dataset info - special note for filtered dataset
        print("📊 Dataset: ChemBL Filtered (Top 10 targets, ≥3 measurements per compound)")
        print("   • 84,413 compounds with high target coverage")
        print("   • 10 most active targets from original 1,310")
        print("   • Quality-focused subset for efficient training")
        
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
        
        # Check if ChemBL-only mode
        if "--chembl_only" in config:
            print(f"🎯 Task: ChemBL classification only (radius-{input_radius} structural input)")
        else:
            print(f"🎯 Task: Predict radius-{target_radius} functional from radius-{input_radius} structural")
        
        print(f"🧪 ChemBL: 10 high-activity targets (no minimum activity filtering needed)")
        
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
        
        # Pretrained model info
        self._print_pretrained_info(config)
        
        print(f"📁 Output: {config['--output_dir']}")
        
        print("=" * 70)
        print()


class FilteredChemBLOnlyConfig(FilteredChemBLConfig):
    """ChemBL-only configuration for filtered dataset (no MLM)."""
    
    def __init__(self):
        super().__init__()
        self.config_name = "filtered_chembl_only"
    
    def get_config(self, overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get filtered ChemBL-only configuration."""
        # Start with base filtered config
        config = super().get_config()
        
        # ChemBL-only specific overrides
        chembl_only_params = {
            "--output_dir": "outputs/chembl_filtered_only",
            "--model_name": "chembl_filtered_only_t4",
            "--batch_size": "32",  # Can use even larger batch without MLM
            "--learning_rate": "2e-4",
            "--warmup_steps": "1000",
            "--max_epochs": "30",
            "--patience": "5",
            "--chembl_only": "",  # Enable ChemBL-only mode
            "--classification_loss_weight": "1.0",
            "--accumulate_grad_batches": "2",
        }
        
        config.update(chembl_only_params)
        if overrides:
            config.update(overrides)
        return config


def run_chembl_training(
    config_class, 
    title: str, 
    overrides: Optional[Dict[str, str]] = None,
    pretrained_path: Optional[str] = None,
    freeze_encoder: bool = False
):
    """
    Common function to run ChemBL training with any configuration.
    
    Args:
        config_class: Configuration class to use (T4Config, T4OnlyConfig, etc.)
        title: Title to display for the training run
        overrides: Optional configuration overrides
        pretrained_path: Optional path to pretrained model checkpoint
        freeze_encoder: Whether to freeze encoder layers (for fine-tuning)
    """
    config_instance = config_class()
    config = config_instance.get_config(overrides)
    
    # Add pretrained model configuration if specified
    if pretrained_path:
        pretrained_config = config_instance.add_pretrained_model(pretrained_path, freeze_encoder)
        config.update(pretrained_config)
    
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