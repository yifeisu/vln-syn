# name of the exp, where the "log.txt" and "weights" store
name=vln_lxmert_4p_envdrop_5m_512_128_5x_wdecay

args="--gpu_id 2
      --name ${name}
      --train auglistener

      --epoch 3
      --batch_size 24
      --optim adamW

      --pretrain_path /data/syf/vln-syn/pretrain_src/snap/pretrain_4p_5m_512_128_5x_envdrop_3gpus/best_model/bert/

      --features clip_vit
      --aug r2r_data/prevalent_aug.json

      --x_layers 5
      --feature_size 512
      --angleFeatSize 128

      --weight_decay 0.0001
      --feedback sample
      --mlWeight 0.20"

nohup python -u r2r_src/train.py $args  > ${name}.log 2>&1 &