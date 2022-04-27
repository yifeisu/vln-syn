import argparse
import os


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")
        # General
        self.parser.add_argument('--name',
                                 type=str,
                                 default='vln_pretrain_3p_5m',
                                 help='experiment id')

        self.parser.add_argument('--epoch',
                                 type=int,
                                 default=2,
                                 help='training epoch')

        self.parser.add_argument('--batchSize',
                                 type=int,
                                 default=12)

        self.parser.add_argument('--lr',
                                 type=float,
                                 default=5e-5,
                                 help="the init learning rate")

        self.parser.add_argument('--gradient_accumulation_steps',
                                 type=int,
                                 default=1)

        self.parser.add_argument("--resume",
                                 type=str,
                                 default=None,
                                 help='path of the trained model to resume')

        self.parser.add_argument('--local_rank',
                                 default=-1,
                                 type=int,
                                 help='node rank for distributed training')

        self.parser.add_argument('--gpu_id',
                                 default=0,
                                 type=str,
                                 help='gpu to used')

        self.parser.add_argument('--seed',
                                 default=0,
                                 type=int)        

        # model architecture
        self.parser.add_argument('--x_layers',
                                 default=5,
                                 type=int)

        self.parser.add_argument('--proxy',
                                 type=str,
                                 default='mlm,nap,tom')

        self.parser.add_argument('--lxmert_pretrain',
                                 type=int,
                                 default=1)

        self.parser.add_argument('--pano',
                                 type=int,
                                 default=0,
                                 help='wether to use pano views in pretraing(nap/nar)')

        # Data preparation
        self.parser.add_argument('--views',
                                 type=int,
                                 default=36,
                                 help='nums of pano image features')

        self.parser.add_argument('--feature',
                                 type=str,
                                 default='clip_vit')

        self.parser.add_argument('--img_feat_dim',
                                 type=int,
                                 default=512)

        self.parser.add_argument('--angle_feat_dim',
                                 type=int,
                                 default=128)

        # Dropout Param
        self.parser.add_argument('--dropout',
                                 type=float,
                                 default=0.5)

        self.parser.add_argument('--featdropout',
                                 type=float,
                                 default=0.3)

        # Training Configurations
        self.parser.add_argument('--num_workers',
                                 type=int,
                                 default=0)

        self.parser.add_argument('--optim',
                                 type=str,
                                 default='adamw')  # rms, adam

        self.parser.add_argument("--betas",
                                 default=[0.9, 0.99],
                                 nargs="+",
                                 help="beta for adam optimizer")

        self.parser.add_argument("--warmup_steps",
                                 default=10000,
                                 type=int,
                                 help="Number of training steps to perform linear learning rate warmup for.")

        self.parser.add_argument('--weight_decay',
                                 dest='weight_decay',
                                 type=float,
                                 default=0.01)

        self.parser.add_argument('--grad_norm',
                                 type=float,
                                 default=10.0)

        self.args = self.parser.parse_args()


param = Param()
args = param.args

args.log_dir = 'snap/%s' % args.name
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
