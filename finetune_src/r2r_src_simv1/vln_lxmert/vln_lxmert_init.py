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

    visual_model = model_class.from_pretrained(args.pretrain_path, config=config)

    return visual_model
