import json
import random
import warnings
import numpy as np
import wandb

import torch
from torch.utils.data import DataLoader
from transformers import LxmertConfig

from data.r2r_dataset import read_img_features, MlmDataset, mlm_collate, NapDataset, nap_collate, TomDataset, tom_collate
from model.pretrain_model import VlnModelPreTraining
from optim.misc import build_optimizer
from optim.sched import get_lr_sched
from utils.logger import LOGGER, print_progress, add_log_to_file
from utils.parameters import args
from utils.validate import validate  # validate_mlm, validate_ncp, validate_nap

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


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    # -------------------------------------------------------------------------------------- #
    # set wandb project
    # -------------------------------------------------------------------------------------- #
    # wandb.init(config=args, project="vln-project-pretrain", entity="susanping")
    #
    # wandb.config.update({
    #     "proxy": ['mlm', 'nap', 'tom'],
    #     "random seed": 0,
    #     "text_model": 'bert',
    #     "vison_model": 'vit',
    #     "xmodal_model": 'lxmert'
    # })

    # ------------------------------------------- #
    # set tensorboard and log writer
    # ------------------------------------------- #
    set_random_seed(0)
    add_log_to_file(LOGGER)

    # ------------------------------------------- #
    # set cuda
    # ------------------------------------------- #
    default_gpu, device, n_gpu = set_cuda()
    if default_gpu:
        LOGGER.info('device: {} n_gpu: {}, 16-bits training: {}'.format(device, n_gpu, bool(0)))

    # ------------------------------------------- #
    # model config and initial or resume the model
    # ------------------------------------------- #
    config = LxmertConfig.from_pretrained('unc-nlp/lxmert-base-uncased')
    config.img_feature_dim = 640
    config.visual_pos_dim = 4
    config.pred_head_dropout_prob = 0.2

    model = VlnModelPreTraining(config=config).to(device)
    if args.resume:
        model.from_pretrained(args.resume)
        LOGGER.info(f"Resume the trained model from {args.resume}")

    # ------------------------------------------- #
    # create dataset and dataloader
    # ------------------------------------------- #
    image_feat = read_img_features('data/CLIP-ViT-B-32-views.tsv', 36)

    # 1.train data
    with open("data/r2r_train.json", 'r') as f:
        train_json_data = json.load(f)

    # with open("data/prevalent_aug.json", 'r') as f:
    #     train_json_data += json.load(f)

    # 2. validate data
    with open("data/r2r_val_unseen.json", 'r') as f:
        val_json_data = json.load(f)

    # with open("data/r2r_val_seen.json", 'r') as f:
    #     val_json_data += json.load(f)

    LOGGER.info(f"Finish loading the json data.")

    # 1.mlm train dataset
    train_mlm_dataset = MlmDataset(train_json_data, image_feat)
    train_mlm_dataloader = DataLoader(train_mlm_dataset,
                                      batch_size=args.batchSize,
                                      shuffle=True,
                                      num_workers=args.num_workers,
                                      collate_fn=mlm_collate,
                                      pin_memory=True,
                                      drop_last=True)
    # 2.nap train dateset
    train_nap_dataset = NapDataset(train_json_data, image_feat)
    train_nap_dataloader = DataLoader(train_nap_dataset,
                                      batch_size=args.batchSize,
                                      shuffle=True,
                                      num_workers=args.num_workers,
                                      collate_fn=nap_collate,
                                      pin_memory=True,
                                      drop_last=True)

    # 3.tom train dateset
    train_tom_dataset = TomDataset(train_json_data, image_feat)
    train_tom_dataloader = DataLoader(train_tom_dataset,
                                      batch_size=args.batchSize,
                                      shuffle=True,
                                      num_workers=args.num_workers,
                                      collate_fn=tom_collate,
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

    LOGGER.info(f"Finish creating all dataset and dataloader, train on {len(train_nap_dataset)} items, validate on {len(val_nap_dataset)} items")
    # ------------------------------------------- #
    # create the optimizer and learning rate
    # ------------------------------------------- #
    # create the loss function
    mlm_loss_fun = torch.nn.CrossEntropyLoss()
    nap_loss_fun = torch.nn.CrossEntropyLoss()
    tom_loss_fun = torch.nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, args)

    # ------------------------------------------- #
    # training and validate process
    # ------------------------------------------- #
    LOGGER.info(f"********** Running training with {n_gpu} GPUs, total epoch {args.epoch}. **********")
    optim_step = 10
    task = args.proxy
    optimizer.zero_grad()
    best_model = {'mlm_acc': 0.0,
                  'nap_acc': 0.0,
                  'tom_acc': 0.0}

    for epoch in range(args.epoch):
        index = 0
        # ------------------------------------------- #
        # training process
        # ------------------------------------------- #
        model.train()
        # obtain the dataloader iter
        train_mlm_iter = iter(train_mlm_dataloader)
        train_nap_iter = iter(train_nap_dataloader)
        train_tom_iter = iter(train_tom_dataloader)

        total_iter = len(train_nap_dataset) // args.batchSize * args.epoch
        warmup_iter = total_iter // 5
        while True:
            index += args.batchSize
            update_para = False

            # 1.train mlm proxy task
            try:
                if 'mlm' in task:
                    data = next(train_mlm_iter)
                    with torch.no_grad():
                        data = [item.to(device) for item in data]

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

            except StopIteration as e:
                print("\nReload the train_mlm_iter, until the train_nap_iter stops itering.")
                train_mlm_iter = iter(train_mlm_dataloader)

            try:
                if 'tom' in task:
                    # 2.train nap proxy task
                    data = next(train_tom_iter)
                    with torch.no_grad():
                        data = [item.to(device) for item in data]

                    instr_ids, instr_mask, pad_traj_views, pad_traj_index, traj_mask, traj_labels = data

                    tom_preds = model('tom',
                                      instr_ids=instr_ids,
                                      instr_mask=instr_mask,
                                      image_feat=(pad_traj_views, pad_traj_index),
                                      image_mask=traj_mask,
                                      teacher_action=traj_labels)

                    tom_loss = tom_loss_fun(tom_preds,
                                            traj_labels)
                    tom_loss.backward()

            except StopIteration as e:
                print("Reload the train_tom_iter, until the train_nap_iter stops itering.")
                train_tom_iter = iter(train_tom_dataloader)

            try:
                if 'nap' in task:
                    # 2.train nap proxy task
                    data = next(train_nap_iter)
                    with torch.no_grad():
                        data = [item.to(device) for item in data]

                    instr_ids, instr_mask, candidate_views, candidate_mask, teacher_action = data

                    nap_preds = model('nap',
                                      instr_ids=instr_ids,
                                      instr_mask=instr_mask,
                                      image_feat=candidate_views,
                                      image_mask=candidate_mask,
                                      teacher_action=teacher_action)

                    nap_loss = nap_loss_fun(nap_preds, teacher_action)
                    nap_loss.backward()

            except StopIteration as e:
                print('\n')
                break

            # total_loss = mlm_loss * 0.4 + tom_loss * 0.3 + nap_loss * 0.6
            # total_loss.backward()

            # 3. update the parameters
            if (index + 1) % args.gradient_accumulation_steps == 0:
                update_para = True
                optim_step += 1

                lr_this_step = get_lr_sched(optim_step, args.lr, warmup_iter, total_iter)
                lr_this_step = max(lr_this_step, 3e-8)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # 4. print the training progress
            print_progress(index, len(train_nap_dataset), prefix='Progress:',
                           suffix='Complete. Mlm loss is %.4f, Nap loss is %.4f, Tom loss is Lr is  %.4f, %.8f, Update para: %s' % (mlm_loss.item(),
                                                                                                                                    nap_loss.item(),
                                                                                                                                    tom_loss.item(),
                                                                                                                                    lr_this_step,
                                                                                                                                    update_para))
            # log with wandb
            # wandb.log({"mlm_loss": mlm_loss.item(),
            #            "nap_loss": nap_loss.item(),
            #            "tom_loss": tom_loss.item(),
            #            "total_loss": total_loss.item(),
            #            "lr": lr_this_step})

            # 5. validate at each log iter
            log_iter = len(train_nap_dataset) // args.batchSize // 4
            if (optim_step % log_iter) == 0:
                if 'mlm' in task:
                    mlm_val = validate(model, 'mlm', val_mlm_dataloader)
                if 'nap' in task:
                    nap_val = validate(model, 'nap', val_nap_dataloader)
                if 'tom' in task:
                    tom_val = validate(model, 'tom', val_tom_dataloader)
                model.train()

                # log with wandb
                # wandb.log({"mlm_acc": mlm_val['acc'],
                #            "nap_acc": nap_val['acc'],
                #            "tom_acc": tom_val['acc']})

                # if mlm_val['acc'] > best_model['mlm_acc'] and nap_val['acc'] > best_model['nap_acc']:
                if (mlm_val['acc'] * 0.4 + nap_val['acc'] * 0.5 + tom_val['acc'] * 0.2) > \
                        (best_model['mlm_acc'] * 0.4 + best_model['nap_acc'] * 0.5 + best_model['tom_acc'] * 0.2):
                    best_model['mlm_acc'], best_model['nap_acc'], best_model['tom_acc'] = mlm_val['acc'], nap_val['acc'], tom_val['acc']

                    save_path = args.log_dir + '/best_model/'
                    model.save_pretrained(save_path)
                    model.bert.save_pretrained(save_path + '/bert')
                    LOGGER.info(f"Best model saved.")

        LOGGER.info(f"Finish the {epoch} train epoch!")

        # ------------------------------------------- #
        # validating process
        # ------------------------------------------- #
        model.eval()
        if 'mlm' in task:
            validate(model, 'mlm', val_mlm_dataloader)
        if 'nap' in task:
            validate(model, 'nap', val_nap_dataloader)
        if 'tom' in task:
            validate(model, 'tom', val_tom_dataloader)
        LOGGER.info(f"Finishs the {epoch} validate epoch!")

        # ------------------------------------------- #
        # save the model
        # ------------------------------------------- #
        save_path = args.log_dir + '/%s' % epoch
        model.save_pretrained(save_path)
        model.bert.save_pretrained(save_path + '/bert')
