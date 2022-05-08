# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

from transformers import BertTokenizer, LxmertConfig
from vln_lxmert.vil_lxmert import VlnModel


def get_tokenizer(args):
    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained("bert-base-uncased")
    return tokenizer


def get_vlnlxmert_models(args, config=None):
    config = LxmertConfig.from_pretrained('unc-nlp/lxmert-base-uncased')
    config.img_feature_type = ""
    config.img_feature_dim = args.feature_size + args.angle_feat_size
    config.visual_pos_dim = 4
    config.x_layers = args.x_layers
    config.r_layers = 0

    model_class = VlnModel
    assert args.pretrain_path, 'you have to provide the pretrained models'

    if args.train == 'validlistener':
        # random initialize the weights
        visual_model = model_class(config=config)
    else:
        if args.pretrain_r2r:
            print('Using our pretrain model.\n')
            visual_model = model_class.from_pretrained(args.pretrain_path, config=config)
        elif args.pretrain_lxmert:
            print('Using the pretrain lxmert model.\n')
            visual_model = model_class.from_pretrained('unc-nlp/lxmert-base-uncased', config=config)
        else:
            print('Random weight.\n')
            visual_model = model_class(config=config)

    return visual_model
