name=lxmert-test

flag="--gpu_id ${1}
      --submit 1
      --test_only 0

      --decision_mode recbert

      --train validlistener
      --resume snap/5-lxmert_4x_5p_places_stop_candidate_eagar_wdecay_cls4nap_recbert4dec_1.75e/state_dict/best_val_unseen

      --pretrain_path snap/vln_pretrain_3p_long

      --features res152-places365
      --feature_size 2048

      --batch_size 8
      --lr 1e-5
      --optim adamW"

python r2r_src/train.py $flag --name $name
