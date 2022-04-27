name=${0}
name=${name%.*}

args="--gpu_id ${1}
      --name ${name}

      --epoch 10
      --batchSize 128
      --num_workers 1

      --weight_decay 0.01
      --gradient_accumulation_steps 4

      --pano 1
      --lxmert_pretrain 0
      --x_layers 4
      --proxy mlm,nap,itm,tom,nar

      --feature res152_places365
      --img_feat_dim 2048
      --angle_feat_dim 128"

nohup python -u -m torch.distributed.launch --nproc_per_node=3 main_r2r_ddp_acc.py $args > ${name}.log 2>&1 &