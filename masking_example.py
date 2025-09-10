#!/usr/bin/env python3
"""
Example usage of the sequence masking functionality for discrete diffusion models.

This script demonstrates how to use the new masking features added to dataloader_with_masking.py
for training discrete diffusion models without clean data.
"""

import os
import torch
from dataloader_with_masking import get_dataset, get_tokenizer, get_dataloaders

class SimpleConfig:
    """Simple configuration class for demonstration."""
    def __init__(self):
        self.data = SimpleNamespace()
        self.loader = SimpleNamespace()
        self.trainer = SimpleNamespace()
        self.model = SimpleNamespace()
        
        # Data configuration
        self.data.train = 'wikitext103'
        self.data.valid = 'wikitext103'
        self.data.cache_dir = './data_cache'
        self.data.wrap = True
        self.data.tokenizer_name_or_path = 'bert-base-uncased'
        
        # Model configuration
        self.model.length = 512
        
        # Loader configuration
        self.loader.batch_size = 8
        self.loader.eval_batch_size = 8
        self.loader.global_batch_size = 8
        self.loader.eval_global_batch_size = 8
        self.loader.num_workers = 2
        self.loader.pin_memory = True
        
        # Trainer configuration
        self.trainer.num_nodes = 1
        self.trainer.accumulate_grad_batches = 1

class SimpleNamespace:
    """Simple namespace for configuration attributes."""
    pass

def demonstrate_masking_functionality():
    """Demonstrate the masking functionality with different configurations."""
    print("Discrete Diffusion Model Dataset Masking Demo")
    print("=" * 50)
    
    # Create configuration
    config = SimpleConfig()
    
    # Get tokenizer
    tokenizer = get_tokenizer(config)
    print(f"Using tokenizer: {type(tokenizer).__name__}")
    print(f"Mask token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
    
    # Example 1: Standard dataset without masking
    print("\n1. Loading standard dataset (no masking):")
    try:
        dataset_clean = get_dataset(
            dataset_name='wikitext103',
            tokenizer=tokenizer,
            wrap=True,
            mode='train',
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            apply_masking=False
        )
        print(f"   Clean dataset size: {len(dataset_clean)}")
        
        # Show a sample
        sample = dataset_clean[0]
        tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'][:20])  # First 20 tokens
        print(f"   Sample (first 20 tokens): {' '.join(tokens)}")
        
    except Exception as e:
        print(f"   Note: Dataset not available ({e})")
    
    # Example 2: Dataset with 90% clean, 10% masked sequences
    print("\n2. Loading dataset with masking (90% clean, 10% masked):")
    try:
        dataset_light_masking = get_dataset(
            dataset_name='wikitext103',
            tokenizer=tokenizer,
            wrap=True,
            mode='train',
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            apply_masking=True,
            clean_ratio=0.9,  # 90% clean sequences
            token_keep_prob=0.5,  # 50% of tokens kept in masked sequences
            masking_seed=42
        )
        print(f"   Lightly masked dataset size: {len(dataset_light_masking)}")
        
        # Show a sample
        sample = dataset_light_masking[0]
        tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'][:20])
        print(f"   Sample (first 20 tokens): {' '.join(tokens)}")
        
    except Exception as e:
        print(f"   Note: Dataset not available ({e})")
    
    # Example 3: Dataset with heavy masking (10% clean, 90% masked)
    print("\n3. Loading dataset with heavy masking (10% clean, 90% masked):")
    try:
        dataset_heavy_masking = get_dataset(
            dataset_name='wikitext103',
            tokenizer=tokenizer,
            wrap=True,
            mode='train',
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            apply_masking=True,
            clean_ratio=0.1,  # Only 10% clean sequences
            token_keep_prob=0.3,  # Only 30% of tokens kept in masked sequences
            masking_seed=42
        )
        print(f"   Heavily masked dataset size: {len(dataset_heavy_masking)}")
        
        # Show a sample
        sample = dataset_heavy_masking[0]
        tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'][:20])
        print(f"   Sample (first 20 tokens): {' '.join(tokens)}")
        
    except Exception as e:
        print(f"   Note: Dataset not available ({e})")

def demonstrate_dataloader_integration():
    """Demonstrate how to integrate masking with the full dataloader pipeline."""
    print("\n" + "=" * 50)
    print("DataLoader Integration Demo")
    print("=" * 50)
    
    config = SimpleConfig()
    tokenizer = get_tokenizer(config)
    
    try:
        # Create dataloaders with masking
        train_loader, valid_loader = get_dataloaders(
            config=config,
            tokenizer=tokenizer,
            skip_train=False,
            skip_valid=True,  # Skip validation for this demo
            apply_masking=True,
            clean_ratio=0.1,  # 10% clean, 90% masked
            token_keep_prob=0.5,  # 50% token keep probability
            masking_seed=42
        )
        
        print(f"Created train loader with {len(train_loader)} batches")
        
        # Show a sample batch
        batch = next(iter(train_loader))
        print(f"Batch shape: {batch['input_ids'].shape}")
        
        # Show first sequence in batch
        first_seq = batch['input_ids'][0]
        tokens = tokenizer.convert_ids_to_tokens(first_seq[:30])  # First 30 tokens
        print(f"First sequence (30 tokens): {' '.join(tokens)}")
        
        # Count masks in the batch
        mask_count = (batch['input_ids'] == tokenizer.mask_token_id).sum().item()
        total_tokens = batch['input_ids'].numel()
        mask_percentage = (mask_count / total_tokens) * 100
        print(f"Masks in batch: {mask_count}/{total_tokens} ({mask_percentage:.1f}%)")
        
    except Exception as e:
        print(f"Note: Full dataloader demo not available ({e})")

def show_masking_parameters_guide():
    """Show a guide for choosing masking parameters."""
    print("\n" + "=" * 50)
    print("Masking Parameters Guide")
    print("=" * 50)
    
    guide = """
    Key Parameters:
    
    1. clean_ratio (float, 0.0-1.0):
       - Fraction of sequences to keep completely clean (unmasked)
       - Example: 0.1 means 10% clean sequences, 90% will be masked
       - Recommended: 0.0-0.2 for training without clean data
       
    2. token_keep_prob (float, 0.0-1.0):
       - For sequences that get masked, probability of keeping each token
       - Example: 0.5 means each token has 50% chance of being kept
       - Recommended: 0.3-0.7 depending on desired difficulty
       
    3. masking_seed (int, optional):
       - Random seed for reproducible masking
       - Use same seed for consistent masking across runs
       
    Example Configurations:
    
    • Light masking (easier): clean_ratio=0.2, token_keep_prob=0.7
      → 20% clean, 80% masked with 70% tokens kept
      
    • Medium masking: clean_ratio=0.1, token_keep_prob=0.5
      → 10% clean, 90% masked with 50% tokens kept
      
    • Heavy masking (harder): clean_ratio=0.0, token_keep_prob=0.3
      → 0% clean, 100% masked with 30% tokens kept
    """
    print(guide)

if __name__ == "__main__":
    demonstrate_masking_functionality()
    demonstrate_dataloader_integration()
    show_masking_parameters_guide()
    
    print("\n" + "=" * 50)
    print("Demo completed! You can now use dataloader_with_masking.py")
    print("for training discrete diffusion models without clean data.")
    print("=" * 50)
