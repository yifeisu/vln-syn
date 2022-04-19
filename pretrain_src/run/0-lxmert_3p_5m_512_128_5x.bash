name=pretrain_3p_5m_2048_128

args="--gpu_id 0
      --name ${name}

      --epoch 4
      --batchSize 64
      --num_workers 12

      --x_layers 5
      --proxy mlm,nap,tom

      --feature clip_vit
      --img_feat_dim 512
      --angle_feat_dim 128"

nohup python -u -m torch.distributed.launch --nproc_per_node=1 main_r2r_3p_ddp.py $args > vln-pretrain.log 2>&1 &