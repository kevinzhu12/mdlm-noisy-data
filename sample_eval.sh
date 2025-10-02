CHECKPOINT_PATH="/home/azureuser/kevin-mdlm/outputs/wikitext103/2025.09.30/dit_small_loglinear_s1_tkp0.1_20250930_013351/checkpoints/best.ckpt"
CACHE_DIR="/home/azureuser/mdlm/data_cache"
DEVICE=${DEVICE:-"cuda:1"}
STEPS=${STEPS:-1000}
SEED=${SEED:-42}      

python main.py \
  mode=sample_eval \
  eval.checkpoint_path="$CHECKPOINT_PATH" \
  data=wikitext103 \
  data.cache_dir="$CACHE_DIR" \
  model.length=512 \
  sampling.predictor=ddpm_cache \
  sampling.steps="$STEPS" \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=1 \
  trainer.devices=1 \
  strategy._target_=lightning.pytorch.strategies.SingleDeviceStrategy \
  ~strategy.find_unused_parameters \
  +strategy.device="$DEVICE" \
  seed="$SEED"