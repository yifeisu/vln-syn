name=pretrain_3p_5m_512_128_5x_envdrop_2gpus

args="--gpu_id 5,6,7
      --name ${name}

      --epoch 4
      --batchSize 48
      --num_workers 12

      --x_layers 5
      --proxy mlm,nap,tom,itm

      --feature clip_vit
      --img_feat_dim 512
      --angle_feat_dim 128"

nohup python -u -m torch.distributed.launch --nproc_per_node=3 main_r2r_ddp.py $args > ${name}.log 2>&1 &