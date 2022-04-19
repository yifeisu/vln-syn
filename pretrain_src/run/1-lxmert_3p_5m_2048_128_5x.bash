gpu_id=0

args="--gpu_id ${gpu_id}
      --name pretrain_3p_5m_2048_128

      --epoch 4
      --batchSize 128
      --num_workers 12

      --feature res152_imagenet

      --img_feat_dim 2048
      --angle_feat_dim 128"

#nohup python -u -m torch.distributed.launch --nproc_per_node=1 main_r2r_3p_ddp.py $args > vln-pretrain.log 2>&1 &
python -u -m torch.distributed.launch --nproc_per_node=1 main_r2r_3p_ddp.py $args