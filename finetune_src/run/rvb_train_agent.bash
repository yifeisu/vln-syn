name=recurrent-vln-bert-prevalent-bs24-imagenet

flag="--vlnbert prevalent

      --aug r2r_data/prevalent_aug.json
      --test_only 0

      --train auglistener

      --features imagenet
      --maxAction 15
      --batchSize 24
      --feedback sample
      --lr 1e-5
      --iters 300000
      --optim adamW

      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python r2r_src/train.py $flag --name $name
