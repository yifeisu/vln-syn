name=pretrain_5p_5m_512_128_4x_envdrop_3gpus_acc8_loader_sample

args="--gpu_id 0,1,2
      --name ${name}

      --epoch 6
      --batchSize 64
      --num_workers 1

      --weight_decay 0.03
      --gradient_accumulation_steps 8

      --lxmert_pretrain 0
      --x_layers 4
      --proxy mlm,nap,itm,tom,nar

      --feature clip_vit
      --img_feat_dim 512
      --angle_feat_dim 128"

nohup python -u -m torch.distributed.launch --nproc_per_node=3 main_r2r_ddp.py $args > ${name}.log 2>&1 &