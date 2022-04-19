name=pretrain_3p_5m_512_128_5x

args="--gpu_id 0
      --name ${name}

      --epoch 3
      --batchSize 64
      --num_workers 1

      --x_layers 5
      --proxy mlm,nap,tom

      --feature clip_vit
      --img_feat_dim 512
      --angle_feat_dim 128"

nohup python -u -m torch.distributed.launch --nproc_per_node=1 main_r2r_ddp.py $args > ${name}.log 2>&1 &