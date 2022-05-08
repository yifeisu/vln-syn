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
	  
	    --pretrain_r2r 0
	    --pretrain_lxmert 1
      --pretrain_path ../pretrain_src/snap/9-lxmert_np_5p_clip_pano_cls4nap_acc1_bs64/1/bert

      --features res152-places365
      --aug r2r_data/prevalent_aug.json
      --speaker_aug 0

      --decision_mode recbert
      --x_layers 4
      --feature_size 2048
      --angleFeatSize 128

      --decay 0.00000001
      --feedback sample
      --mlWeight 0.20"

nohup python -u r2r_src_simv1/train.py $args  > ${name}.log 2>&1 &