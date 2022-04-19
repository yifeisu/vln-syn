from utils.parameters import args
import os

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
import json
import random
import warnings
import numpy as np
import wandb

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from transformers import LxmertConfig

from data.r2r_dataset import read_img_features, MlmDataset, mlm_collate, NapDataset, nap_collate, TomDataset, tom_collate, ItmDataset, itm_collate
from model.pretrain_model import VlnModelPreTraining
from optim.misc import build_optimizer
from optim.sched import get_lr_sched
from utils.logger import LOGGER, print_progress, add_log_to_file
from utils.validate import validate

warnings.filterwarnings("ignore")


def set_cuda():
    """
    Initialize CUDA for distributed computing
    """
    if not torch.cuda.is_available():
        return True, torch.device("cpu"), 0

    _default_gpu = True
    _device = torch.device("cuda")
    _n_gpu = torch.cuda.device_count()

    return _default_gpu, _device, _n_gpu


def set_random_seed(_seed):
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)


if __name__ == '__main__':
    # ------------------------------------------- #
    # set seed and log writer
    # ------------------------------------------- #
    seed = args.seed
    if args.local_rank != -1:
        seed += args.local_rank
    set_random_seed(seed)
    add_log_to_file(LOGGER)

    # ------------------------------------------- #
    # set cuda and distributed training
    # ------------------------------------------- #
    # init the ddp
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    default_gpu, device, n_gpu = set_cuda()
    if default_gpu:
        LOGGER.info('device: {} n_gpu: {}, 16-bits training: {}'.format(device, args.local_rank, bool(0)))

    # ------------------------------------------- #
    # model config and initial or resume the model
    # ------------------------------------------- #
    config = LxmertConfig.from_pretrained('unc-nlp/lxmert-base-uncased')
    config.img_feature_dim = args.img_feat_dim + args.angle_feat_dim
    config.visual_pos_dim = 4
    config.x_layers = args.x_layers
    config.pretrain_tasks = args.proxy
    config.pred_head_dropout_prob = 0.2

    model = VlnModelPreTraining(config=config).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    if args.resume:
        model.module.from_pretrained(args.resume)
        LOGGER.info(f"Resume the trained model from {args.resume}")

    # ------------------------------------------- #
    # create dataset and dataloader
    # ------------------------------------------- #
    if args.feature == 'clip_vit':
        image_feat = read_img_features('data/CLIP-ViT-B-32-views.tsv', 36)
    elif args.feature == 'res152_imagenet':
        image_feat = read_img_features('data/ResNet-152-imagenet.tsv', 36)

    # 1.train data
    with open("data/r2r_train.json", 'r') as f:
        train_json_data = json.load(f)

    with open("data/prevalent_aug.json", 'r') as f:
        train_json_data += json.load(f)

    # 2. validate data
    with open("data/r2r_val_seen.json", 'r') as f:
        val_json_data = json.load(f)

    with open("data/r2r_val_unseen.json", 'r') as f:
        val_json_data += json.load(f)

    LOGGER.info(f"Finish loading the json data.")

    # 1.mlm train dataset
    train_mlm_dataset = MlmDataset(train_json_data, image_feat)
    train_mlm_sampler = DistributedSampler(train_mlm_dataset)
    train_mlm_dataloader = DataLoader(train_mlm_dataset,
                                      batch_size=args.batchSize,
                                      num_workers=args.num_workers,
                                      collate_fn=mlm_collate,
                                      sampler=train_mlm_sampler,
                                      pin_memory=True,
                                      drop_last=True)
    # 2.nap train dateset
    train_nap_dataset = NapDataset(train_json_data, image_feat)
    train_nap_sampler = DistributedSampler(train_nap_dataset)
    train_nap_dataloader = DataLoader(train_nap_dataset,
                                      batch_size=args.batchSize,
                                      num_workers=args.num_workers,
                                      collate_fn=nap_collate,
                                      sampler=train_nap_sampler,
                                      pin_memory=True,
                                      drop_last=True)

    # 3.tom train dateset
    train_tom_dataset = TomDataset(train_json_data, image_feat)
    train_tom_sampler = DistributedSampler(train_tom_dataset)
    train_tom_dataloader = DataLoader(train_tom_dataset,
                                      batch_size=args.batchSize,
                                      num_workers=args.num_workers,
                                      collate_fn=tom_collate,
                                      sampler=train_tom_sampler,
                                      pin_memory=True,
                                      drop_last=True)

    # 4.itm train dateset
    train_itm_dataset = ItmDataset(train_json_data, image_feat)
    train_itm_sampler = DistributedSampler(train_itm_dataset)
    train_itm_dataloader = DataLoader(train_itm_dataset,
                                      batch_size=args.batchSize,
                                      num_workers=args.num_workers,
                                      collate_fn=itm_collate,
                                      sampler=train_itm_sampler,
                                      pin_memory=True,
                                      drop_last=True)

    # 1.mlm val dataset
    val_mlm_dataset = MlmDataset(val_json_data, image_feat)
    val_mlm_dataloader = DataLoader(val_mlm_dataset,
                                    batch_size=args.batchSize,
                                    collate_fn=mlm_collate,
                                    drop_last=True)
    # 2.nap val dateset
    val_nap_dataset = NapDataset(val_json_data, image_feat)
    val_nap_dataloader = DataLoader(val_nap_dataset,
                                    batch_size=args.batchSize,
                                    collate_fn=nap_collate,
                                    drop_last=True)

    # 3.tom val dateset
    val_tom_dataset = TomDataset(val_json_data, image_feat)
    val_tom_dataloader = DataLoader(val_tom_dataset,
                                    batch_size=args.batchSize,
                                    collate_fn=tom_collate,
                                    drop_last=True)

    # 4.itm val dateset
    val_itm_dataset = ItmDataset(val_json_data, image_feat)
    val_itm_dataloader = DataLoader(val_itm_dataset,
                                    batch_size=args.batchSize,
                                    collate_fn=itm_collate,
                                    drop_last=True)

    LOGGER.info(f"Finish creating all dataset and dataloader, train on {len(train_nap_dataloader.sampler)} items, validate on {len(val_nap_dataset)} items")

    # ------------------------------------------- #
    # create loss function and the optimizer
    # ------------------------------------------- #
    # create the loss function
    mlm_loss_fun = torch.nn.CrossEntropyLoss()
    nap_loss_fun = torch.nn.CrossEntropyLoss()
    tom_loss_fun = torch.nn.CrossEntropyLoss()
    itm_loss_fun = torch.nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, args)

    # ------------------------------------------- #
    # training and validate process
    # ------------------------------------------- #
    LOGGER.info(f"********** Running training with {args.local_rank} GPU, total epoch {args.epoch}. **********")
    optim_step = 10
    optimizer.zero_grad()
    best_model = {'score': 0.0,
                  'mlm_acc': 0.0,
                  'nap_acc': 0.0,
                  'tom_acc': 0.0,
                  'itm_acc': 0.0}

    for epoch in range(args.epoch):
        train_mlm_sampler.set_epoch(epoch)
        train_nap_sampler.set_epoch(epoch)
        train_tom_sampler.set_epoch(epoch)
        train_itm_sampler.set_epoch(epoch)
        # obtain the dataloader iter
        train_mlm_iter = iter(train_mlm_dataloader)
        train_nap_iter = iter(train_nap_dataloader)
        train_tom_iter = iter(train_tom_dataloader)
        train_itm_iter = iter(train_itm_dataloader)

        # ------------------------------------------- #
        # training process
        # ------------------------------------------- #
        index = 0
        model.train()

        total_iter = len(train_nap_dataloader.sampler) // args.batchSize * args.epoch
        warmup_iter = total_iter // 5
        while True:
            # -------------------------------------------------------------------------------------- #
            # set wandb project
            # -------------------------------------------------------------------------------------- #
            if args.local_rank == 0 and index == args.batchSize:
                wandb.init(config=args, project="vln-project-pretrain", entity="susanping")

                wandb.config.update({
                    "proxy": args.proxy,
                    "text_model": 'bert',
                    "vison_model": args.feature,
                    "xmodal_model": 'lxmert'
                })

            index += args.batchSize
            # 1.train mlm proxy task
            if 'mlm' in args.proxy:
                try:
                    data = next(train_mlm_iter)
                except StopIteration as e:
                    print("\nReload the train_mlm_iter, until the train_nap_iter stops itering.")
                    train_mlm_iter = iter(train_mlm_dataloader)
                    data = next(train_mlm_iter)

                with torch.no_grad():
                    data = [item.cuda(non_blocking=True) for item in data]

                instr_ids, instr_labels, instr_mask, pad_traj_views, traj_mask = data

                mlm_preds = model('mlm',
                                  instr_ids=instr_ids,
                                  instr_labels=instr_labels,
                                  instr_mask=instr_mask,
                                  image_feat=pad_traj_views,
                                  image_mask=traj_mask)

                mlm_loss = mlm_loss_fun(mlm_preds,
                                        instr_labels[instr_labels != -1])
                mlm_loss.backward()

            # 2.train tom proxy task
            if 'tom' in args.proxy:
                try:
                    data = next(train_tom_iter)
                except StopIteration as e:
                    print("Reload the train_tom_iter, until the train_nap_iter stops itering.")
                    train_tom_iter = iter(train_tom_dataloader)
                    data = next(train_tom_iter)

                with torch.no_grad():
                    data = [item.cuda(non_blocking=True) for item in data]

                instr_ids, instr_mask, pad_traj_views, pad_traj_index, traj_mask, traj_labels = data

                tom_pred = model('tom',
                                 instr_ids=instr_ids,
                                 instr_mask=instr_mask,
                                 image_feat=(pad_traj_views, pad_traj_index),
                                 image_mask=traj_mask,
                                 teacher_action=traj_labels)

                tom_loss = tom_loss_fun(tom_pred,
                                        traj_labels)
                tom_loss.backward()

            # 3.train itm proxy task
            if 'itm' in args.proxy:
                try:
                    data = next(train_itm_iter)
                except StopIteration as e:
                    print("Reload the train_itm_iter, until the train_nap_iter stops itering.")
                    train_itm_iter = iter(train_itm_dataloader)
                    data = next(train_itm_iter)

                with torch.no_grad():
                    data = [item.cuda(non_blocking=True) for item in data]

                instr_ids, instr_mask, pad_traj_views, traj_mask, traj_labels = data

                itm_pred = model('itm',
                                 instr_ids=instr_ids,
                                 instr_mask=instr_mask,
                                 image_feat=pad_traj_views,
                                 image_mask=traj_mask,
                                 teacher_action=traj_labels)

                itm_loss = itm_loss_fun(itm_pred,
                                        traj_labels)
                itm_loss.backward()

            # 4.train nap proxy task
            if 'nap' in args.proxy:
                try:
                    data = next(train_nap_iter)
                except StopIteration as e:
                    print('\n')
                    break

                with torch.no_grad():
                    data = [item.cuda(non_blocking=True) for item in data]

                instr_ids, instr_mask, candidate_views, candidate_mask, teacher_action = data

                nap_preds = model('nap',
                                  instr_ids=instr_ids,
                                  instr_mask=instr_mask,
                                  image_feat=candidate_views,
                                  image_mask=candidate_mask,
                                  teacher_action=teacher_action)

                nap_loss = nap_loss_fun(nap_preds, teacher_action)
                nap_loss.backward()

            # 3. update the parameters
            if (index + 1) % args.gradient_accumulation_steps == 0:
                optim_step += 1

                lr_this_step = get_lr_sched(optim_step, args.lr, warmup_iter, total_iter)
                lr_this_step = max(lr_this_step, 3e-8)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # 4. print the training progress
            if index < 1500:
                loss_str = ''
                if 'mlm' in args.proxy:
                    loss_str += 'mlm loss %.4f,' % mlm_loss.item()
                if 'tom' in args.proxy:
                    loss_str += 'tom loss %.4f,' % tom_loss.item()
                if 'nap' in args.proxy:
                    loss_str += 'nap loss %.4f,' % nap_loss.item()
                if 'itm' in args.proxy:
                    loss_str += 'itm loss %.4f,' % itm_loss.item()

                loss_str += 'lr is  %.4f, device id: %s.' % (lr_this_step, args.local_rank)
                print_progress(index, len(train_nap_dataloader.sampler), prefix='Progress:', suffix='Complete. %s' % loss_str)

            # log with wandb
            if args.local_rank == 0 and index > args.batchSize:
                if 'mlm' in args.proxy:
                    wandb.log({"mlm_loss": mlm_loss.item()})
                if 'tom' in args.proxy:
                    wandb.log({"tom_loss": tom_loss.item()})
                if 'nap' in args.proxy:
                    wandb.log({"nap_loss": nap_loss.item()})
                if 'itm' in args.proxy:
                    wandb.log({"itm_loss": itm_loss.item()})

                wandb.log({"lr": lr_this_step})
                wandb.log({"optim_step": optim_step})

            # 5. validate at each log iter
            val_iter = len(train_nap_dataloader.sampler) // args.batchSize // 4
            if args.local_rank == 0:
                if (optim_step % val_iter) == 0:
                    print("validate for best model", args.proxy)
                    now_model = {'score': 0.0, 'mlm_acc': 0.0, 'nap_acc': 0.0, 'tom_acc': 0.0, 'itm_acc': 0.0}
                    if 'mlm' in args.proxy:
                        mlm_val = validate(model, 'mlm', val_mlm_dataloader)
                        wandb.log({"mlm_acc": mlm_val['acc']})
                        now_model['score'] += mlm_val['acc'] * 0.5
                        now_model['mlm_acc'] = mlm_val['acc']

                    if 'nap' in args.proxy:
                        nap_val = validate(model, 'nap', val_nap_dataloader)
                        wandb.log({"nap_acc": nap_val['acc']})
                        now_model['score'] += nap_val['acc'] * 0.5
                        now_model['nap_acc'] = nap_val['acc']

                    if 'tom' in args.proxy:
                        tom_val = validate(model, 'tom', val_tom_dataloader)
                        wandb.log({"tom_acc": tom_val['acc']})
                        now_model['score'] += tom_val['acc'] * 0.3
                        now_model['tom_acc'] = tom_val['acc']

                    if 'itm' in args.proxy:
                        itm_val = validate(model, 'itm', val_itm_dataloader)
                        wandb.log({"itm_acc": itm_val['acc']})
                        now_model['score'] += itm_val['acc'] * 0.4
                        now_model['itm_acc'] = itm_val['acc']

                    model.train()

                    # record the best model
                    if now_model['score'] > best_model['score']:
                        best_model.update(now_model)

                        save_path = args.log_dir + '/best_model/'
                        model.module.save_pretrained(save_path)
                        model.module.bert.save_pretrained(save_path + '/bert')
                        LOGGER.info(f"Best model saved.")

        LOGGER.info(f"Finish the {epoch} train epoch!")

        # ------------------------------------------- #
        # validating process
        # ------------------------------------------- #
        if args.local_rank == 0:
            model.eval()
            if 'mlm' in args.proxy:
                validate(model, 'mlm', val_mlm_dataloader)
            if 'nap' in args.proxy:
                validate(model, 'nap', val_nap_dataloader)
            if 'tom' in args.proxy:
                validate(model, 'tom', val_tom_dataloader)
            if 'itm' in args.proxy:
                validate(model, 'itm', val_itm_dataloader)
            LOGGER.info(f"Finishs the {epoch} validate epoch!")

            # ------------------------------------------- #
            # save the model
            # ------------------------------------------- #
            save_path = args.log_dir + '/%s' % epoch
            model.module.save_pretrained(save_path)
            model.module.bert.save_pretrained(save_path + '/bert')

    dist.barrier()
