import argparse
import os

import torch


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # General
        self.parser.add_argument('--test_only',
                                 type=int,
                                 default=0,
                                 help='fast mode for testing')

        self.parser.add_argument('--name',
                                 type=str,
                                 default='vln_lxmert_no_name',
                                 help='experiment id')

        self.parser.add_argument("--submit",
                                 type=int,
                                 default=0)
                                 
        # Model architecture
        self.parser.add_argument("--x_layers",
                                 type=int,
                                 default=5)

        # Training process hyperparameters
        self.parser.add_argument('--gpu_id',
                                 default=0,
                                 type=str,
                                 help='gpu to used')

        self.parser.add_argument('--train',
                                 type=str,
                                 default='auglistener')

        self.parser.add_argument('--epoch',
                                 type=int,
                                 default=5,
                                 help='training epochs')

        self.parser.add_argument('--batch_size',
                                 type=int,
                                 default=8)

        self.parser.add_argument('--optim',
                                 type=str,
                                 default='adamW')  # rms, adam

        self.parser.add_argument('--lr',
                                 type=float,
                                 default=1e-5,
                                 help="the learning rate")

        self.parser.add_argument('--decay',
                                 dest='weight_decay',
                                 type=float,
                                 default=0.0003)

        self.parser.add_argument("--resume",
                                 default=None,
                                 help='path of the trained model')

        self.parser.add_argument("--pretrain_path",
                                 type=str,
                                 default=None)

        self.parser.add_argument('--dropout',
                                 type=float,
                                 default=0.5)

        self.parser.add_argument('--featdropout',
                                 type=float,
                                 default=0.4)

        # Augmented Paths
        self.parser.add_argument("--features",
                                 type=str,
                                 default='clip_vit',
                                 help='clip_vit, res152-imagenet')

        self.parser.add_argument("--aug",
                                 default='r2r_data/prevalent_aug.json')

        self.parser.add_argument("--speaker_aug",
                                 type=int,
                                 default=0)

        # Data preparation
        self.parser.add_argument('--maxInput',
                                 type=int,
                                 default=80,
                                 help="max input length of the instructions")

        self.parser.add_argument('--maxAction',
                                 type=int,
                                 default=15,
                                 help='Max Action sequence')

        self.parser.add_argument('--feature_size',
                                 type=int,
                                 default=512,
                                 help='size of the precomputed image features')

        self.parser.add_argument("--angleFeatSize",
                                 dest="angle_feat_size",
                                 type=int,
                                 default=128,
                                 help='angle feature size')

        self.parser.add_argument('--ignoreid',
                                 type=int,
                                 default=-100)

        self.parser.add_argument("--loadOptim",
                                 action="store_const",
                                 default=False,
                                 const=True)

        # Listener Model Config
        self.parser.add_argument("--zeroInit",
                                 dest='zero_init',
                                 action='store_const',
                                 default=False,
                                 const=True)

        self.parser.add_argument("--mlWeight",
                                 dest='ml_weight',
                                 type=float,
                                 default=0.20)

        self.parser.add_argument("--teacherWeight",
                                 dest='teacher_weight',
                                 type=float,
                                 default=1.0)

        # Training Configurations

        self.parser.add_argument('--feedback',
                                 type=str,
                                 default='sample',
                                 help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')

        self.parser.add_argument('--teacher',
                                 type=str,
                                 default='final',
                                 help="How to get supervision. one of ``next`` and ``final`` ")

        self.parser.add_argument('--epsilon',
                                 type=float,
                                 default=0.01)

        # A2C
        self.parser.add_argument("--gamma",
                                 default=0.9,
                                 type=float)
        self.parser.add_argument("--normalize",
                                 dest="normalize_loss",
                                 default="total",
                                 type=str,
                                 help='batch or total')

        self.args = self.parser.parse_args()

        if self.args.optim == 'rms':
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            self.args.optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            self.args.optimizer = torch.optim.SGD
        else:
            assert False


param = Param()
args = param.args

# -------------------------------------------------------------------------------------- #
# make the logdir
# -------------------------------------------------------------------------------------- #
args.log_dir = 'snap/%s/' % args.name
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
