#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

now=$(date +"%Y%m%d_%H%M%S")
echo 'log_time:' $now

export CUDA_VISIBLE_DEVICES=0,1,2
python -m torch.distributed.launch --master_port 1237 --nproc_per_node=3 --use_env \
         train.py  --config ${config} --log_time $now
