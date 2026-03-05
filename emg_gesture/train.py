#!/usr/bin/env python3
"""
Training Entry Point for EMG Gesture Classification.

Trains the BiLSTM-CNN hybrid model on recorded EMG data.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent))

import torch
import yaml

from data.dataset import create_dataloaders, generate_mock_recordings
from model.bilstm_cnn import create_model_from_config
from model.trainer import Trainer


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description='Train EMG Gesture Classification Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--recordings', type=str, default=None,
        help='Path to recordings directory (overrides config)'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--generate-mock', action='store_true',
        help='Generate mock training data before training'
    )
    parser.add_argument(
        '--mock-samples', type=int, default=5000,
        help='Number of mock samples per class'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        choices=['cpu', 'cuda', 'mps'],
        help='Device to train on (auto-detected if not specified)'
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply command-line overrides
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr

    # Save updated config temporarily if needed
    temp_config_path = config_path
    if args.epochs or args.batch_size or args.lr:
        temp_config_path = Path('_temp_config.yaml')
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)

    print("=" * 60)
    print("EMG Gesture Classification Training")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Gesture classes: {config['gesture_classes']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print("=" * 60)

    # Generate mock data if requested
    if args.generate_mock:
        print("\nGenerating mock training data...")
        generate_mock_recordings(
            str(temp_config_path),
            n_samples_per_class=args.mock_samples,
            output_dir=args.recordings
        )

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, dataset = create_dataloaders(
        str(temp_config_path),
        recordings_dir=args.recordings,
        compute_stats=True
    )

    if train_loader is None or len(dataset) == 0:
        print("\nError: No training data available!")
        print("Use --generate-mock to create synthetic training data, or")
        print("record real data using: python -m data.collector --mock --record <label>")
        sys.exit(1)

    # Print data statistics
    print(f"\nDataset statistics:")
    distribution = dataset.get_class_distribution()
    for gesture, count in distribution.items():
        print(f"  {gesture}: {count} samples")

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Create model
    print("\nCreating model...")
    model = create_model_from_config(str(temp_config_path))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Get class weights if configured
    class_weights = None
    if config['training'].get('class_weights'):
        class_weights = torch.tensor(config['training']['class_weights'])
        print(f"Using class weights: {class_weights}")
    else:
        # Use computed weights for imbalanced data
        class_weights = dataset.get_class_weights()
        print(f"Computed class weights: {class_weights}")

    # Create trainer and train
    trainer = Trainer(model, str(temp_config_path), device=device)

    print("\nStarting training...")
    results = trainer.train(
        train_loader,
        val_loader,
        class_weights=class_weights,
        resume=args.resume
    )

    # Print final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Epochs trained: {results['epochs_trained']}")
    print(f"\nModel saved to: {config['paths']['checkpoints_dir']}/best_model.pt")
    print(f"Training log saved to: {config['paths']['training_log']}")

    # Cleanup temp config
    if temp_config_path != config_path and temp_config_path.exists():
        temp_config_path.unlink()


if __name__ == '__main__':
    main()
