# name of the exp, where the "log.txt" and "weights" store

args="--name vln_lxmert_3p_more
      --train auglistener

      --epoch 3
      --batch_size 8
      --optim adamW
      --lr 1e-5

      --pretrain_path snap/vln_pretrain_3p_long

      --features clip_vit
      --aug r2r_data/prevalent_aug.json

      --feature_size 512
      --angleFeatSize 128

      --feedback sample
      --mlWeight 0.20"


python r2r_src/train.py $args