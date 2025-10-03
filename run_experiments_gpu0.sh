#!/bin/bash

# Run experiment 1: clean data
echo "Starting experiment 1: Clean data"
python main.py \
  model=small \
  data=wikitext103 \
  data.cache_dir=/home/azureuser/kevin-mdlm/data_cache2 \
  parameterization=subs \
  model.length=512 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  trainer.devices=1 \
  trainer.accumulate_grad_batches=2 \
  strategy._target_=lightning.pytorch.strategies.SingleDeviceStrategy \
  ~strategy.find_unused_parameters \
  +strategy.device=cuda:0 \
  loader.global_batch_size=256 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  premasked_training.apply_masking=False \
  trainer.val_check_interval=500 \
  trainer.max_steps=6500 \
  wandb.name=clean_baseline

# Run experiment 2: masking_prob=0.1
echo "Starting experiment 2: masking_prob=0.1"
python main.py \
  model=small \
  data=wikitext103 \
  data.cache_dir=/home/azureuser/kevin-mdlm/data_cache2 \
  parameterization=subs \
  model.length=512 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  trainer.devices=1 \
  trainer.accumulate_grad_batches=2 \
  strategy._target_=lightning.pytorch.strategies.SingleDeviceStrategy \
  ~strategy.find_unused_parameters \
  +strategy.device=cuda:0 \
  loader.global_batch_size=256 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  premasked_training.apply_masking=True \
  premasked_training.masking_prob=0.1 \
  trainer.val_check_interval=500 \
  trainer.max_steps=6500 

# Run experiment 3: masking_prob=0.3
echo "Starting experiment 3: masking_prob=0.3"
python main.py \
  model=small \
  data=wikitext103 \
  data.cache_dir=/home/azureuser/kevin-mdlm/data_cache2 \
  parameterization=subs \
  model.length=512 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  trainer.devices=1 \
  trainer.accumulate_grad_batches=2 \
  strategy._target_=lightning.pytorch.strategies.SingleDeviceStrategy \
  ~strategy.find_unused_parameters \
  +strategy.device=cuda:0 \
  loader.global_batch_size=256 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  premasked_training.apply_masking=True \
  premasked_training.masking_prob=0.3 \
  trainer.val_check_interval=500 \
  trainer.max_steps=6500

# Run experiment 4: masking_prob=0.7, clean_ratio=0.1
echo "Starting experiment 4: masking_prob=0.7, clean_ratio=0.1"
python main.py \
  model=small \
  data=wikitext103 \
  data.cache_dir=/home/azureuser/kevin-mdlm/data_cache2 \
  parameterization=subs \
  model.length=512 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  trainer.devices=1 \
  trainer.accumulate_grad_batches=2 \
  strategy._target_=lightning.pytorch.strategies.SingleDeviceStrategy \
  ~strategy.find_unused_parameters \
  +strategy.device=cuda:0 \
  loader.global_batch_size=256 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  premasked_training.apply_masking=True \
  premasked_training.masking_prob=0.7 \
  premasked_training.clean_ratio=0.1 \
  trainer.val_check_interval=500 \
  trainer.max_steps=6500

# Run experiment 5: masking_prob=0.9, clean_ratio=0.1
echo "Starting experiment 5: masking_prob=0.9, clean_ratio=0.1"
python main.py \
  model=small \
  data=wikitext103 \
  data.cache_dir=/home/azureuser/kevin-mdlm/data_cache2 \
  parameterization=subs \
  model.length=512 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  trainer.devices=1 \
  trainer.accumulate_grad_batches=2 \
  strategy._target_=lightning.pytorch.strategies.SingleDeviceStrategy \
  ~strategy.find_unused_parameters \
  +strategy.device=cuda:0 \
  loader.global_batch_size=256 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  premasked_training.apply_masking=True \
  premasked_training.masking_prob=0.9 \
  premasked_training.clean_ratio=0.1 \
  trainer.val_check_interval=500 \
  trainer.max_steps=6500


echo "All experiments completed!"