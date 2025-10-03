# #!/usr/bin/env python3
# """
# Script to visualize dataset masking for debugging
# """
import torch
import transformers
from datasets import load_dataset
import dataloader
import random
import functools
import os
import utils
import datasets
import re
import itertools
import hydra

LOGGER = utils.get_logger(__name__)

def wt_detokenizer(string):
  # contractions
  string = string.replace("s '", "s'")
  string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
  # number separators
  string = string.replace(" @-@ ", "-")
  string = string.replace(" @,@ ", ",")
  string = string.replace(" @.@ ", ".")
  # punctuation
  string = string.replace(" : ", ": ")
  string = string.replace(" ; ", "; ")
  string = string.replace(" . ", ". ")
  string = string.replace(" ! ", "! ")
  string = string.replace(" ? ", "? ")
  string = string.replace(" , ", ", ")
  # double brackets
  string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
  string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
  string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
  string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
  string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
  # miscellaneous
  string = string.replace("= = = =", "====")
  string = string.replace("= = =", "===")
  string = string.replace("= =", "==")
  string = string.replace(" " + chr(176) + " ", chr(176))
  string = string.replace(" \n", "\n")
  string = string.replace("\n ", "\n")
  string = string.replace(" N ", " 1 ")
  string = string.replace(" 's", "'s")
  return string

def _group_texts(examples, block_size, bos, eos):
  # Concatenate all texts.
  concatenated_examples = list(itertools.chain(* examples['input_ids']))
  total_length = len(concatenated_examples)
  # TODO(yair): look into not dropping the remainder but rather padding it.
  # We drop the small remainder, and if the total_length < block_size - 2
  # we exclude this batch and return an empty dict.
  # We could add padding if the model supported it instead of
  # this drop, you can customize this part to your needs.
  new_block_size = block_size - 2  # [BOS] and [EOS] to be added
  total_length = (total_length // new_block_size) * new_block_size
  # Split by chunks of max_len.
  result = {}
  _values = []
  _attn_masks = []
  for i in range(0, total_length, new_block_size):
    _values.append(
      [bos]
      + concatenated_examples[i : i + new_block_size]
      + [eos])
    _attn_masks.append(torch.ones(block_size))
  result['input_ids'] = _values
  result['attention_mask'] = _attn_masks
  return result


def apply_sequence_masking(dataset, tokenizer, clean_ratio=0.2, masking_prob=0.5, seed=None):
  """
  Apply masking to a dataset for training discrete diffusion models without clean data.

  Args:
    dataset: HuggingFace dataset with 'input_ids' column containing tokenized sequences (shape: [B, sequence_length])
    tokenizer: Tokenizer with mask_token attribute
    clean_ratio: Fraction of sequences to keep clean (default: 0.2 for 20% clean)
    masking_prob: Probability of masking each token in masked sequences (default: 0.5)
    seed: Random seed for reproducibility (optional)

  Returns:
    Modified dataset with masking applied
  """

  if seed is not None:
    random.seed(seed)

  token_keep_prob = 1 - masking_prob

  if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
    mask_token_id = tokenizer.mask_token_id
  else:
    mask_token_id = tokenizer.encode(tokenizer.mask_token)[0]
  
  def mask_sequence(example):
    input_ids = example['input_ids']

    if isinstance(input_ids[0], list):
      masked_input_ids = []
      for seq in input_ids:
        if random.random() < clean_ratio:
          masked_input_ids.append(seq)
        else:
          masked_seq = []
          for token_id in seq:
            if token_id == tokenizer.bos_token_id or token_id == tokenizer.eos_token_id or token_id == tokenizer.pad_token_id:
              masked_seq.append(token_id)
            else:
              if random.random() < token_keep_prob:
                masked_seq.append(token_id)
              else:
                masked_seq.append(mask_token_id)
          masked_input_ids.append(masked_seq)
      example['input_ids'] = masked_input_ids
    else:
      if random.random() < clean_ratio:
        pass
      else:
        masked_seq = []
        for token_id in input_ids:
          if token_id == tokenizer.bos_token_id or token_id == tokenizer.eos_token_id or token_id == tokenizer.pad_token_id:
            masked_seq.append(token_id)
          else:
            if random.random() < token_keep_prob:
              masked_seq.append(token_id)
            else:
              masked_seq.append(mask_token_id)
        example['input_ids'] = masked_seq
    
    return example
  
  masked_dataset = dataset.map(mask_sequence, batched=True, desc='Applying sequence masking')
  return masked_dataset

def get_dataset(
    dataset_name, tokenizer, wrap, mode, cache_dir,
    block_size=1024, num_proc=len(os.sched_getaffinity(0)), streaming=False,
    apply_masking=False, clean_ratio=0.2, masking_prob=0.5, masking_seed=None):
  if wrap:
    filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped.dat'
  else:
    filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped.dat'
  _path = os.path.join(cache_dir, filename)
  
  if utils.fsspec_exists(_path):
    LOGGER.info(f'Loading data from: {_path}')
    return datasets.load_from_disk(_path).with_format('torch')
  LOGGER.info(f'Generating new data at: {_path}')

  crop_train = dataset_name == 'text8-crop'
  if mode == 'train' and crop_train:
    # double block size for sub-sampling
    block_size *= 2
  
  if dataset_name == 'wikitext103':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-103-raw-v1',
      cache_dir=cache_dir)
  elif dataset_name == 'wikitext2':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-2-raw-v1',
      cache_dir=cache_dir)
  elif dataset_name == 'ptb':
    dataset = datasets.load_dataset(
      'ptb_text_only', cache_dir=cache_dir)
  elif dataset_name == 'lambada':
    dataset = get_lambada_test_dataset()
  elif dataset_name == 'text8':
    assert wrap
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size)
  elif dataset_name == 'text8-crop':
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size, crop_train=True)
  elif dataset_name == 'openwebtext-train':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[:-100000]',
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'openwebtext-valid':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[-100000:]',
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'scientific_papers_arxiv':
    dataset = datasets.load_dataset(
      'scientific_papers', 'arxiv',
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'scientific_papers_pubmed':
    dataset = datasets.load_dataset(
      'scientific_papers', 'pubmed',
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'ag_news':
    dataset = datasets.load_dataset(
      'ag_news',
      cache_dir=cache_dir,
      streaming=streaming)
  else:
    dataset = datasets.load_dataset(
      dataset_name,
      cache_dir=cache_dir,
      streaming=streaming)

  if dataset_name in ['lambada', 'openwebtext-train',
                      'openwebtext-valid']:
    data = dataset
  else:
    data = dataset[mode]

  if dataset_name.startswith('wikitext'):
    detokenizer = wt_detokenizer
  elif dataset_name == 'ptb':
    detokenizer = ptb_detokenizer
  elif dataset_name == 'lm1b':
    detokenizer = lm1b_detokenizer
  elif dataset_name == 'lambada':
    detokenizer = lambada_detokenizer
  elif dataset_name.startswith('scientific_papers'):
    detokenizer = scientific_papers_detokenizer
  else:
    detokenizer = None

  def _apply_detokenizer(detokenizer):
    def detok(text):
      for i, t in enumerate(text, 0):
        text[i] = detokenizer(t)
      return text
    return detok
  
  EOS = tokenizer.encode(tokenizer.eos_token)[0]
  BOS = tokenizer.encode(tokenizer.bos_token)[0]

  def preprocess_and_tokenize(example):
    if dataset_name == 'ptb':
      text = example['sentence']
    elif 'scientific_papers' in dataset_name:
      text = example['article']
    else:
      text = example['text']
    
    if detokenizer is not None:
      text = _apply_detokenizer(detokenizer)(text)

    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    if wrap:
      tokens = tokenizer(text,
                         add_special_tokens=False,
                         return_attention_mask=False,
                         return_token_type_ids=False)
      tokens = {'input_ids':
                [t + [EOS] for t in tokens['input_ids']]}
      # Still missing BOS, but will be added in group_texts
    else:
      tokens = tokenizer(text,
                         max_length=block_size,
                         padding='max_length',
                         truncation=True,
                         add_special_tokens=True,
                         return_attention_mask=True,
                         return_token_type_ids=True)
    return tokens

  if streaming:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      desc='Tokenizing')
  else:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Tokenizing')
  if dataset_name == 'ptb':
    tokenized_dataset = tokenized_dataset.remove_columns(
      'sentence')
  elif 'scientific_papers' in dataset_name:
    tokenized_dataset = tokenized_dataset.remove_columns([
      'article', 'abstract', 'section_names'])
  elif dataset_name == 'ag_news':
    tokenized_dataset = tokenized_dataset.remove_columns(
      ['text', 'label'])
  else:
    tokenized_dataset = tokenized_dataset.remove_columns(
      'text')

  if not wrap:
    if apply_masking:
      tokenized_dataset = apply_sequence_masking(
        tokenized_dataset, tokenizer, clean_ratio, masking_prob, masking_seed)
    tokenized_dataset.save_to_disk(_path)
    return tokenized_dataset.with_format('torch')

  group_texts = functools.partial(
    _group_texts, block_size=block_size, bos=BOS, eos=EOS)
  if streaming:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      desc='Grouping')
  else:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Grouping')
  
  if apply_masking:
    chunked_dataset = apply_sequence_masking(
      chunked_dataset, tokenizer, clean_ratio, masking_prob, masking_seed)
  
  if not streaming:
    chunked_dataset.save_to_disk(_path)
  chunked_dataset = chunked_dataset.with_format('torch')
  return chunked_dataset

@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
    """Main entry point for training."""
    # L.seed_everything(config.seed)
    # _print_config(config, resolve=True, save_cfg=True)
    
    # visualize_masking()

    # dataset = load_dataset(
    #   'wikitext',
    #   name='wikitext-103-raw-v1',
    #   cache_dir='./data_cache')
    
    tokenizer = dataloader.get_tokenizer(config)
    # tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.pad_token_id = tokenizer.eos_token_id

    # masked_dataset = apply_sequence_masking(dataset, tokenizer, clean_ratio=0.2, masking_prob=0.5, seed=42)

    train_set = get_dataset(
      'wikitext103',
      tokenizer,
      mode='train',
      wrap=True,
      cache_dir='./data_cache',
      apply_masking=True,
      clean_ratio=0.01,
      masking_prob=0.5,
      masking_seed=42
    )

    # print(train_set[0][])
    print()
    print(tokenizer.decode(train_set[0]['input_ids']))

if __name__ == "__main__":
    main()