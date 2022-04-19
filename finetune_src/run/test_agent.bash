name=lxmert-test

flag="--submit 1
      --test_only 0

      --train validlistener
      --resume snap/vln_lxmert_3p_long/state_dict/best_val_unseen

      --pretrain_path snap/vln_pretrain_3p_long


      --features clip_vit
      --batch_size 8
      --lr 1e-5
      --optim adamW"

python r2r_src/train.py $flag --name $name
