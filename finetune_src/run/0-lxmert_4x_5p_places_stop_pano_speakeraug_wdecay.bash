# name of the exp, where the "log.txt" and "weights" store
name=${0}
name=${name%.*}
name=${name#*/}

args="--gpu_id ${1}
      --name ${name}
      --train auglistener

      --epoch 4
      --batch_size 16
      --optim adamW

      --pretrain_path ../pretrain_src/snap/0-lxmert_4x_scratch_5p_places_envdrop_pano_acc/best_model_1/bert

      --features res152-places365
      --aug r2r_data/prevalent_aug.json
      --speaker_aug 1

      --x_layers 4
      --feature_size 2048
      --angleFeatSize 128

      --decay 0.00001
      --feedback sample
      --mlWeight 0.20"

nohup python -u r2r_src/train.py $args  > ${name}.log 2>&1 &