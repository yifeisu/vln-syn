# name of the exp, where the "log.txt" and "weights" store
name=vln_lxmert_5p_envdrop_5m_512_128_4x_wdecay_no_lxpre_acc8_loadersample_deletecliptok_3eb

args="--gpu_id 3
      --name ${name}
      --train auglistener

      --epoch 4
      --batch_size 16
      --optim adamW

      --pretrain_path ../pretrain_src/snap/pretrain_5p_5m_512_128_4x_envdrop_3gpus_acc8_loader_sample/2/bert/

      --features clip_vit
      --aug r2r_data/prevalent_aug.json

      --x_layers 4
      --feature_size 512
      --angleFeatSize 128

      --decay 0.000003
      --feedback sample
      --mlWeight 0.20"

nohup python -u r2r_src/train.py $args  > ${name}.log 2>&1 &