# Run experiment 1: masking_prob=0.7
echo "Starting experiment 1: masking_prob=0.7"
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
  +strategy.device=cuda:1 \
  loader.global_batch_size=256 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  premasked_training.apply_masking=True \
  premasked_training.masking_prob=0.7 \
  trainer.val_check_interval=500 \
  trainer.max_steps=6500 

# Run experiment 2: masking_prob=0.9
echo "Starting experiment 2: masking_prob=0.9"
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
  +strategy.device=cuda:1 \
  loader.global_batch_size=256 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  premasked_training.apply_masking=True \
  premasked_training.masking_prob=0.9 \
  trainer.val_check_interval=500 \
  trainer.max_steps=6500

# Run experiment 3: masking_prob=0.1, clean_ratio=0.1
echo "Starting experiment 3: masking_prob=0.1, clean_ratio=0.1"
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
  +strategy.device=cuda:1 \
  loader.global_batch_size=256 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  premasked_training.apply_masking=True \
  premasked_training.masking_prob=0.1 \
  premasked_training.clean_ratio=0.1 \
  trainer.val_check_interval=500 \
  trainer.max_steps=6500

# Run experiment 4: masking_prob=0.3, clean_ratio=0.1
echo "Starting experiment 4: masking_prob=0.3, clean_ratio=0.1"
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
  +strategy.device=cuda:1 \
  loader.global_batch_size=256 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  premasked_training.apply_masking=True \
  premasked_training.masking_prob=0.3 \
  premasked_training.clean_ratio=0.1 \
  trainer.val_check_interval=500 \
  trainer.max_steps=6500

# Run experiment 5: masking_prob=0.5, clean_ratio=0.1
echo "Starting experiment 5: masking_prob=0.5, clean_ratio=0.1"
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
  +strategy.device=cuda:1 \
  loader.global_batch_size=256 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  premasked_training.apply_masking=True \
  premasked_training.masking_prob=0.5 \
  premasked_training.clean_ratio=0.1 \
  trainer.val_check_interval=500 \
  trainer.max_steps=6500

echo "All experiments completed!"