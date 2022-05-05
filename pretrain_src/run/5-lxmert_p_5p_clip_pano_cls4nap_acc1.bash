name=${0}
name=${name%.*}
name=${name#*/}

args="--gpu_id ${1}
      --name ${name}

      --epoch 10
      --batchSize 128
      --num_workers 1

      --grad_norm 5.0
      --weight_decay 0.01
      --gradient_accumulation_steps 1

      --nap_mode cls
      --pano 1
      --lxmert_pretrain 1
      --x_layers 4
      --proxy mlm,nap,itm,tom,nar

      --feature clip_vit
      --img_feat_dim 512
      --angle_feat_dim 128"

nohup python -u -m torch.distributed.launch --nproc_per_node=4 main_r2r_ddp_acc.py $args > ${name}.log 2>&1 &