name=pretrain_4p_5m_512_128_4x_envdrop_3gpus_no_lxpre_loader_sample

args="--gpu_id 0,1,2
      --name ${name}

      --epoch 4
      --batchSize 128
      --num_workers 1

      --gradient_accumulation_steps 1

      --lxmert_pretrain 0
      --x_layers 4
      --proxy mlm,nap,tom,itm

      --feature clip_vit
      --img_feat_dim 512
      --angle_feat_dim 128"

nohup python -u -m torch.distributed.launch --nproc_per_node=3 main_r2r_ddp.py $args > ${name}.log 2>&1 &