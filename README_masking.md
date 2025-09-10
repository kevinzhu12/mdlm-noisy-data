# Dataset Masking for Discrete Diffusion Models

This implementation adds sequence masking functionality to train discrete diffusion models without clean data, as requested for the research project.

## Files Added

- `dataloader_with_masking.py` - Enhanced dataloader with masking functionality
- `test_masking.py` - Test script demonstrating the masking features
- `masking_example.py` - Comprehensive example and usage guide

## Key Features

### Sequence Masking Logic

The implementation provides a function `apply_sequence_masking()` that:

1. **Selects sequences for masking**: Based on `clean_ratio` parameter
   - `clean_ratio=0.1` → 10% sequences stay clean, 90% get masked
2. **Token-level masking**: For sequences selected for masking
   - Each token has `token_keep_prob` chance of staying unmasked
   - `token_keep_prob=0.5` → 50% probability each token stays original
3. **Preserves special tokens**: BOS, EOS, and PAD tokens are never masked

### Usage Examples

```python
from dataloader_with_masking import get_dataset, get_tokenizer

# Get tokenizer
config = YourConfig()
tokenizer = get_tokenizer(config)

# Load dataset with masking
dataset = get_dataset(
    dataset_name='wikitext103',
    tokenizer=tokenizer,
    wrap=True,
    mode='train',
    cache_dir='./data_cache',
    block_size=512,
    apply_masking=True,
    clean_ratio=0.1,        # 10% clean sequences
    token_keep_prob=0.5,    # 50% token keep probability
    masking_seed=42         # For reproducibility
)
```

### Integration with DataLoaders

```python
from dataloader_with_masking import get_dataloaders

train_loader, valid_loader = get_dataloaders(
    config=config,
    tokenizer=tokenizer,
    apply_masking=True,
    clean_ratio=0.1,
    token_keep_prob=0.5,
    masking_seed=42
)
```

## Parameter Recommendations

| Use Case       | clean_ratio | token_keep_prob | Description                |
| -------------- | ----------- | --------------- | -------------------------- |
| Light masking  | 0.2         | 0.7             | 20% clean, easier training |
| Medium masking | 0.1         | 0.5             | 10% clean, balanced        |
| Heavy masking  | 0.0         | 0.3             | 0% clean, challenging      |

## Example Output

Original sequence:

```
[CLS] hello world this is a test sentence [SEP]
```

With masking (clean_ratio=0.0, token_keep_prob=0.5):

```
[CLS] [MASK] world [MASK] [MASK] a [MASK] sentence [SEP]
```

## Testing

Run the test scripts to verify functionality:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run basic tests
python test_masking.py

# Run comprehensive examples
python masking_example.py
```

## Implementation Details

- **Caching**: Masked datasets are cached separately with parameters in filename
- **Reproducibility**: Use `masking_seed` for consistent results
- **Performance**: Masking is applied efficiently using HuggingFace datasets `.map()`
- **Compatibility**: Works with all existing tokenizers and datasets

This implementation enables training discrete diffusion models on datasets where the model never observes completely clean sequences, as specified in the research requirements.
