#!/usr/bin/env python3
"""
Script to visualize and compare the generated masked datasets.
"""

import os
import datasets
from transformers import AutoTokenizer
import random

def load_dataset_from_cache(cache_path):
    """Load a cached dataset."""
    if os.path.exists(cache_path):
        return datasets.load_from_disk(cache_path)
    else:
        print(f"Dataset not found: {cache_path}")
        return None

def analyze_masking_statistics(dataset, tokenizer):
    """Analyze masking statistics in a dataset."""
    if dataset is None:
        return None
    
    mask_token_id = tokenizer.mask_token_id
    total_tokens = 0
    masked_tokens = 0
    clean_sequences = 0
    partially_masked_sequences = 0
    
    # Sample a subset for analysis (first 1000 examples)
    sample_size = min(1000, len(dataset))
    
    for i in range(sample_size):
        example = dataset[i]
        input_ids = example['input_ids']
        
        # Count tokens and masks
        seq_total = len(input_ids)
        seq_masked = sum(1 for token_id in input_ids if token_id == mask_token_id)
        
        total_tokens += seq_total
        masked_tokens += seq_masked
        
        # Classify sequence
        if seq_masked == 0:
            clean_sequences += 1
        else:
            partially_masked_sequences += 1
    
    mask_percentage = (masked_tokens / total_tokens * 100) if total_tokens > 0 else 0
    clean_seq_percentage = (clean_sequences / sample_size * 100) if sample_size > 0 else 0
    
    return {
        'sample_size': sample_size,
        'total_tokens': total_tokens,
        'masked_tokens': masked_tokens,
        'mask_percentage': mask_percentage,
        'clean_sequences': clean_sequences,
        'partially_masked_sequences': partially_masked_sequences,
        'clean_seq_percentage': clean_seq_percentage
    }

def visualize_sequences(dataset, tokenizer, num_examples=5, max_length=100):
    """Show example sequences from the dataset."""
    if dataset is None:
        return
    
    print("Example sequences:")
    print("-" * 80)
    
    # Get random examples
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    
    for i, idx in enumerate(indices):
        example = dataset[idx]
        input_ids = example['input_ids']
        
        # Convert to list if it's a tensor
        if hasattr(input_ids, 'tolist'):
            input_ids = input_ids.tolist()
        
        # Truncate if too long
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            truncated = "... (truncated)"
        else:
            truncated = ""
        
        # Debug: Check for problematic token IDs
        vocab_size = tokenizer.vocab_size
        problematic_ids = [tid for tid in input_ids if tid >= vocab_size or tid < 0]
        if problematic_ids:
            print(f"Warning: Found token IDs outside vocab range: {problematic_ids[:5]}...")
        
        # Convert to tokens - handle None values safely
        try:
            tokens = []
            for j, token_id in enumerate(input_ids):
                if token_id >= vocab_size or token_id < 0:
                    tokens.append(f"[ID:{token_id}]")
                else:
                    token = tokenizer.convert_ids_to_tokens([token_id])[0]
                    if token is None:
                        tokens.append(f"[ID:{token_id}]")
                    else:
                        tokens.append(token)
        except Exception as e:
            print(f"Error converting tokens for example {i+1}: {e}")
            tokens = [f"[ID:{token_id}]" for token_id in input_ids]
        
        # Count masks in this sequence
        mask_count = sum(1 for token in tokens if token == tokenizer.mask_token)
        
        # Count mask token IDs directly as backup
        mask_id_count = sum(1 for token_id in input_ids if token_id == tokenizer.mask_token_id)
        
        print(f"Example {i+1} (Index {idx}):")
        print(f"  Length: {len(input_ids)} tokens, Masks: {mask_count} (ID count: {mask_id_count})")
        print(f"  Token IDs (first 10): {input_ids[:10]}")
        print(f"  Tokens: {' '.join(tokens)}{truncated}")
        print()

def debug_tokenizer_vocab(tokenizer):
    """Debug tokenizer vocabulary issues."""
    print("Tokenizer Debug Info:")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Mask token: '{tokenizer.mask_token}' (ID: {tokenizer.mask_token_id})")
    print(f"  BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f"  EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"  PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"  UNK token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
    print()

def compare_datasets():
    """Compare all generated datasets."""
    print("Dataset Visualization and Comparison")
    print("=" * 60)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print(f"Using tokenizer: {tokenizer.__class__.__name__}")
    print(f"Mask token: '{tokenizer.mask_token}' (ID: {tokenizer.mask_token_id})")
    
    # Add debug info
    debug_tokenizer_vocab(tokenizer)
    
    # Define datasets to compare
    base_path = "/home/azureuser/mdlm/data_cache"
    datasets_to_check = [
        ("Original (Clean)", "wikitext103_train_bs512_wrapped.dat"),
        ("Light Masking (90% clean)", "wikitext103_train_bs512_wrapped_masked_clean0.9_keep0.5_seed42.dat"),
        ("Heavy Masking (10% clean)", "wikitext103_train_bs512_wrapped_masked_clean0.1_keep0.3_seed42.dat"),
    ]
    
    loaded_datasets = {}
    
    # Load and analyze each dataset
    for name, folder in datasets_to_check:
        full_path = os.path.join(base_path, folder)
        print(f"Loading {name}...")
        print(f"Path: {full_path}")
        
        dataset = load_dataset_from_cache(full_path)
        if dataset is not None:
            loaded_datasets[name] = dataset
            print(f"  ✓ Loaded successfully: {len(dataset)} examples")
            
            # Analyze statistics
            stats = analyze_masking_statistics(dataset, tokenizer)
            if stats:
                print(f"  📊 Statistics (from {stats['sample_size']} examples):")
                print(f"     • Total tokens: {stats['total_tokens']:,}")
                print(f"     • Masked tokens: {stats['masked_tokens']:,} ({stats['mask_percentage']:.1f}%)")
                print(f"     • Clean sequences: {stats['clean_sequences']} ({stats['clean_seq_percentage']:.1f}%)")
                print(f"     • Partially masked: {stats['partially_masked_sequences']}")
        else:
            print(f"  ✗ Failed to load")
        print()
    
    # Show example sequences from each dataset
    print("\n" + "=" * 60)
    print("EXAMPLE SEQUENCES")
    print("=" * 60)
    
    for name, dataset in loaded_datasets.items():
        print(f"\n{name.upper()}")
        print("=" * len(name))
        visualize_sequences(dataset, tokenizer, num_examples=3, max_length=50)

if __name__ == "__main__":
    # Set random seed for reproducible examples
    random.seed(42)
    
    # Run visualization
    compare_datasets()
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)