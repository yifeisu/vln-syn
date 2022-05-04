# name of the exp, where the "log.txt" and "weights" store
name=${0}
name=${name%.*}
name=${name#*/}

args="--gpu_id ${1}
      --name ${name}
      --train auglistener

      --epoch 4
      --batch_size 8
      --optim adamW

      --pretrain_path ../pretrain_src/snap/3-lxmert_4x_scratch_5p_places_envdrop_pano_cls4nap_acc1_11111/best_model_4/bert

      --features res152-places365
      --aug r2r_data/prevalent_aug.json
      --speaker_aug 0

      --decision_mode recbert
      --x_layers 4
      --feature_size 2048
      --angleFeatSize 128

      --decay 0.0000001
      --feedback sample
      --mlWeight 0.20"

nohup python -u r2r_src_simv1/train.py $args  > ${name}.log 2>&1 &