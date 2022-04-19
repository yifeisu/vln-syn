# name of the exp, where the "log.txt" and "weights" store
name=vln_lxmert_3p_5m_2048_512
args="--name ${name}
      --train auglistener

      --epoch 3
      --batch_size 16
      --optim adamW

      --pretrain_path snap/vln_pretrain_3p_long

      --features res152-imagenet
      --aug r2r_data/prevalent_aug.json

      --feature_size 2048
      --angleFeatSize 128

      --feedback sample
      --mlWeight 0.20"

#CUDA_VISIBLE_DEVICES = 0
#python r2r_src/train.py $args
nohup python -u r2r_src/train.py $args  > ${name}.log 2>&1 &