#CUDA_VISIBLE_DIVICES=0

args="--name pretrain_3p_5m
      --epoch 2
      --batchSize 12
      --num_workers 0"

python -m torch.distributed.launch --nproc_per_node=1 main_r2r_ddp.py $args