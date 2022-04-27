# name of the exp, where the "log.txt" and "weights" store
name=${0}
name=${name%.*}

args="--gpu_id ${1}
      --name ${name}
      --train auglistener

      --epoch 4
      --batch_size 16
      --optim adamW

      --pretrain_path ../pretrain_src/snap/pretrain_5p_5m_512_128_4x_envdrop_stop_eager_with_pretrain_pano/1/bert

      --features clip_vit
      --aug r2r_data/prevalent_aug.json
      --speaker_aug 1

      --x_layers 4
      --feature_size 2048
      --angleFeatSize 128

      --decay 0.00001
      --feedback sample
      --mlWeight 0.20"

nohup python -u r2r_src/train.py $args  > ${name}.log 2>&1 &