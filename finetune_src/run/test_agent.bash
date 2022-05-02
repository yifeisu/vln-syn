name=lxmert-test

flag="--gpu_id ${1}
      --submit 1
      --test_only 0

      --decision_mode recbert

      --train validlistener
      --resume snap/6-lxmert_4x_3p_places_stop_pano_eagar_wdecay_2e_cls4nap_recbert4dec_4e/state_dict/best_val_unseen

      --pretrain_path snap/vln_pretrain_3p_long

      --features res152-places365
      --feature_size 2048

      --batch_size 8
      --lr 1e-5
      --optim adamW"

python r2r_src_simv1/train.py $flag --name $name
