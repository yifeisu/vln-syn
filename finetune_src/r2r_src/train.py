from param import args
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

import json
import random
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import wandb

from agent import Seq2SeqAgent
from env import R2RBatch
from eval import Evaluation

from utils import timeSince, read_img_features, print_progress
from vln_lxmert.vln_lxmert_init import get_tokenizer

warnings.filterwarnings("ignore")

print(args)


# -------------------------------------------------------------------------------------- #
# train the listener
# -------------------------------------------------------------------------------------- #
def train(train_env, tok, n_iters, log_every=1000, val_envs={}, aug_env=None):
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    # init the wandb writer and record experiment args
    record_file = open(args.log_dir + 'log.txt', 'a')
    record_file.write(str(args) + '\n\n')
    record_file.close()

    # option for resume the train process
    start_iter = 0
    if args.resume is not None:
        if args.aug is None:
            start_iter = listner.load(os.path.join(args.resume))
            print("\nNo Aug LOAD the model from {}, iteration {}".format(args.resume, start_iter))
        else:
            load_iter = listner.load(os.path.join(args.resume))
            print("\nAug LOAD the model from {}, iteration {}".format(args.resume, load_iter))

    start = time.time()
    print('\nListener training starts, start iteration: %s, total iterations: %s' % (str(start_iter), n_iters))

    best_val = {'val_unseen': {"spl": 0.,
                               "sr": 0.,
                               "state": "",
                               'update': False}}

    # -------------------------------------------------------------------------------------- #
    # Run train
    # -------------------------------------------------------------------------------------- #
    for idx in range(start_iter, start_iter + n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters - idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=args.feedback)  # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                args.ml_weight = 0.2
                listner.train(1, feedback=args.feedback)

                # Train with Augmented data
                listner.env = aug_env
                args.ml_weight = 0.2
                listner.train(1, feedback=args.feedback)

                loss_log_str = ", the IL loss is %.4f." % listner.logs['IL_loss'][-1]
                print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete' + loss_log_str)

        if idx == start_iter:
            # some key args to record
            wandb.init(config=args, project="vln-project-finetune", entity="susanping")

        # Log the training stats to wandb
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['critic_loss']), 1)
        critic_loss = sum(listner.logs['critic_loss']) / total
        RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
        IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
        entropy = sum(listner.logs['entropy']) / total

        if idx >= start_iter:
            wandb.log({"train/critic": critic_loss,
                       "train/IL_loss": IL_loss,
                       "train/RL_loss": RL_loss,
                       "train/total_actions": total})

        # writer.add_scalar("policy_entropy", entropy, idx)
        # writer.add_scalar("max_length", length, idx)

        # -------------------------------------------------------------------------------------- #
        # Run validation
        # -------------------------------------------------------------------------------------- #
        loss_str = "iter {}".format(iter)
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            result = listner.get_results()
            score_summary, _ = evaluator.score(result)
            loss_str += ", %s " % env_name
            for metric, val in score_summary.items():
                if metric in ['spl']:
                    # choose the best model according to the spl metric
                    # writer.add_scalar("spl/%s" % env_name, val, idx)
                    wandb.log({"%s/spl" % env_name: val})

                    if env_name in best_val:
                        if val > best_val[env_name]['spl']:
                            best_val[env_name]['spl'] = val
                            best_val[env_name]['update'] = True
                        elif (val == best_val[env_name]['spl']) and (score_summary['success_rate'] > best_val[env_name]['sr']):
                            best_val[env_name]['spl'] = val
                            best_val[env_name]['update'] = True

                if metric in ['success_rate', 'oracle_rate']:
                    # record the success rate and oracle_rate
                    wandb.log({"%s/%s" % (env_name, metric): val})

                loss_str += ', %s: %.4f' % (metric, val)

        record_file = open(args.log_dir + 'log.txt', 'a')
        record_file.write(loss_str + '\n\n')
        record_file.close()

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % env_name))
            else:
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "latest_dict"))

        print('%s (%d %d%%) %s' % (timeSince(start, float(iter) / n_iters),
                                   iter,
                                   float(iter) / n_iters * 100,
                                   loss_str))

        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])
                record_file = open(args.log_dir + 'log.txt', 'a')
                record_file.write('BEST RESULT TILL NOW: ' + env_name + ' | ' + best_val[env_name]['state'] + '\n\n\n')
                record_file.close()

    listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % idx))


def valid(train_env, tok, val_envs={}):
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent.load(args.resume), args.resume))

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        result = agent.get_results()

        if env_name != '':
            score_summary, _ = evaluator.score(result)
            loss_str = "Env name: %s" % env_name
            for metric, val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

        if args.submit:
            json.dump(
                result,
                open(os.path.join(args.log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )


def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(0)
    np.random.seed(0)


def train_val(test_only=False):
    """ Train on the training set, and validate on seen and unseen splits. """
    setup()
    tok = get_tokenizer(args)

    feat_dict = read_img_features(args.features, test_only=test_only)

    if test_only:
        featurized_scans = None
        val_env_names = ['val_train_seen']
    else:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']

    train_env = R2RBatch(feat_dict, batch_size=args.batch_size, splits=['train'], tokenizer=tok)
    from collections import OrderedDict

    if args.submit:
        val_env_names.append('test')
    else:
        pass

    val_envs = OrderedDict(((split,
                             (R2RBatch(feat_dict, batch_size=args.batch_size, splits=[split], tokenizer=tok),
                              Evaluation([split], featurized_scans, tok)))
                            for split in val_env_names))

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validlistener':
        valid(train_env, tok, val_envs=val_envs)
    else:
        assert False


def train_val_augment(test_only=False):
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    tok_bert = get_tokenizer(args)

    # Load the env pre-trained img features
    feat_dict = read_img_features(args.features, test_only=test_only)

    if test_only:
        featurized_scans = None
        val_env_names = ['val_train_seen']
    else:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']

    # Create the training and aug training environment
    train_env = R2RBatch(feat_dict,
                         batch_size=args.batch_size,
                         splits=['train'],
                         tokenizer=tok_bert)

    aug_env = R2RBatch(feat_dict,
                       batch_size=args.batch_size,
                       splits=[args.aug],
                       tokenizer=tok_bert,
                       name='aug')

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batch_size, splits=[split], tokenizer=tok_bert),
                        Evaluation([split], featurized_scans, tok_bert)) for split in val_env_names}

    # compute the total iters
    iters = (len(train_env) + len(aug_env)) * args.epoch
    # Start training
    train(train_env, tok_bert, iters, val_envs=val_envs, aug_env=aug_env)


if __name__ == "__main__":
    # -------------------------------------------------------------------------------------- #
    # init the wandb project and runs
    # -------------------------------------------------------------------------------------- #

    if args.train in ['listener', 'validlistener']:
        train_val(test_only=args.test_only)
    elif args.train == 'auglistener':
        train_val_augment(test_only=args.test_only)
    else:
        assert False

    wandb.finish()
