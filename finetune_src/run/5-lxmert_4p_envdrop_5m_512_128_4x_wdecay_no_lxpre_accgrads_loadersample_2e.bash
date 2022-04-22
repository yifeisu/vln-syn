# name of the exp, where the "log.txt" and "weights" store
name=vln_lxmert_4p_envdrop_5m_512_128_4x_wdecay_no_lxpre_accgrdas_loadersample_2e_deletecliptok

args="--gpu_id 3
      --name ${name}
      --train auglistener

      --epoch 4
      --batch_size 24
      --optim adamW

      --pretrain_path ../pretrain_src/snap/pretrain_4p_5m_512_128_4x_envdrop_3gpus_accgrads_loader_sample/1/bert/

      --features clip_vit
      --aug r2r_data/prevalent_aug.json

      --x_layers 4
      --feature_size 512
      --angleFeatSize 128

      --decay 0.000001
      --feedback sample
      --mlWeight 0.20"

nohup python -u r2r_src/train.py $args  > ${name}.log 2>&1 &