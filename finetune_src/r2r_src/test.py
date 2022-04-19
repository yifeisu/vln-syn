from param import args
from vln_lxmert.vln_lxmert_init import get_vlnlxmert_models

if __name__ == '__main__':
    args.pretrain_path = 'snap/vlnmodel/bert/'
    args.angle_feat_size = 0
    vlnmodel = get_vlnlxmert_models(args, config=None)

