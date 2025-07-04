#!/usr/bin/env python3
"""
Training script with comprehensive gradient and NaN monitoring using 
PyTorch Lightning's built-in capabilities.
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("🔍 Enhanced Monitoring Mode")
    print("=" * 50)
    
    # Conservative parameters with enhanced monitoring
    params = {
        "--train_data": "data/guacamol_v1_all.smiles",
        "--input_vocab": "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        "--target_vocab": "mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
        "--output_dir": "outputs/monitored_training",
        "--model_name": "monitored_training_v1",
        # Conservative model settings
        "--hidden_size": "256",
        "--num_hidden_layers": "6",
        "--num_attention_heads": "4",
        "--intermediate_size": "1024",
        "--dropout": "0.1",
        # Environment configuration
        "--input_radius": "0",
        "--target_radius": "1",
        "--target_use_features": "",
        # Conservative training
        "--batch_size": "16",
        "--learning_rate": "2e-5",
        "--weight_decay": "0.01",
        "--warmup_steps": "5000",
        "--max_epochs": "10",
        "--validation_split": "0.05",
        "--val_check_interval": "0.25",
        # Hardware settings
        "--gpus": "1",
        "--num_workers": "2",
        "--precision": "32",  # Full precision for stability
        "--accumulate_grad_batches": "2",
        "--gradient_clip_val": "0.5",
        "--max_length": "100",
        "--seed": "42",
        # ENHANCED MONITORING FLAGS
        "--track_grad_norm": "",  # Track gradient norms
        "--log_predictions": "",
        "--enable_checkpointing": "",
        "--early_stop_patience": "5",
        "--monitor": "val_loss",
        "--save_top_k": "3",
        # Lightning specific monitoring
        "--detect_anomaly": "",  # PyTorch anomaly detection
        "--enable_progress_bar": "",
        "--log_every_n_steps": "10",
        # Professional logging settings
        "--default_root_dir": "outputs/monitored_training",
        "--enable_model_summary": "",
        "--profiler": "simple",  # Enable simple profiler
    }

    # Build command
    script_path = Path(__file__).parent.parent / "mole" / "cli" / "train_crossenv_mlm.py"
    cmd = [sys.executable, str(script_path)]
    
    for key, value in params.items():
        cmd.append(key)
        if value:
            cmd.append(value)
    
    # Add any additional arguments from command line
    cmd.extend(sys.argv[1:])

    print("🚀 Enhanced Monitoring Training:")
    print(f"  🏗️  Model: {params['--hidden_size']} hidden, {params['--num_hidden_layers']} layers")
    print(f"  💾 Batch: {params['--batch_size']}")
    print(f"  📚 LR: {params['--learning_rate']}")
    print(f"  🛡️  Clip: {params['--gradient_clip_val']}")
    print(f"  ⚡ Precision: {params['--precision']}-bit")
    print("=" * 50)
    print()
    
    print("🔍 Monitoring Features Enabled:")
    print("  • Gradient norm tracking")
    print("  • PyTorch anomaly detection")
    print("  • Model profiling")
    print("  • Detailed checkpointing")
    print("  • Prediction logging")
    print("  • Progress monitoring")
    print()

    try:
        print("🚀 Starting enhanced monitoring training...")
        
        # Set environment variables for additional monitoring
        import os
        env = os.environ.copy()
        env['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous CUDA for better error reporting
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Better memory management
        
        result = subprocess.run(cmd, check=True, env=env)
        print("✅ Training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        return e.returncode
        
    except KeyboardInterrupt:
        print("⏹️  Training interrupted by user")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code if exit_code else 0) 