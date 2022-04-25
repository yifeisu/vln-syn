# name of the exp, where the "log.txt" and "weights" store

# add the stop action, with lxmert pretrain, train with eager mode.
name=vln_lxmert_5p_envdrop_5m_512_128_4x_wdecay_stop_with_pretrain_eager

args="--gpu_id 3
      --name ${name}
      --train auglistener

      --epoch 4
      --batch_size 16
      --optim adamW

      --pretrain_path ../pretrain_src/snap/pretrain_5p_5m_512_128_4x_envdrop_stop_3gpus_eager_with_pretrain/1/bert/

      --features clip_vit
      --aug r2r_data/prevalent_aug.json

      --x_layers 4
      --feature_size 512
      --angleFeatSize 128

      --decay 0.000003
      --feedback sample
      --mlWeight 0.20"

nohup python -u r2r_src/train.py $args  > ${name}.log 2>&1 &