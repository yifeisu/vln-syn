CUDA_VISIBLE_DIVICES=0,1

args="--name pretrain_3p_5m_2048_128
      --epoch 4
      --batchSize 128
      --num_workers 12

      --feature res152_imagenet

      --img_feat_dim 2048
      --angle_feat_dim 128"

# python -m torch.distributed.launch --nproc_per_node=4 main_r2r_ddp.py $args

nohup python -u -m torch.distributed.launch --nproc_per_node=4 main_r2r_3p_ddp.py $args > vln-pretrain.log 2>&1 &