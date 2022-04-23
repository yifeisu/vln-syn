import time
import torch
import torch.nn.functional as F

from .logger import LOGGER


def validate(_model, _task, val_dataloader, setname=''):
    _model.eval()
    if _task == 'mlm':
        val_log = validate_mlm(_model, val_dataloader)
    elif _task == 'nap':
        val_log = validate_nap(_model, val_dataloader)
    elif _task == 'nar':
        val_log = validate_nar(_model, val_dataloader)
    elif _task == 'tom':
        val_log = validate_tom(_model, val_dataloader)
    elif _task == 'itm':
        val_log = validate_itm(_model, val_dataloader)
    else:
        raise ValueError(f'Undefined task {_task}')
    return val_log
    # val_log = {f'val{setname}_{_task}_{k}': v for k, v in val_log.items()}


@torch.no_grad()
def validate_mlm(_model, val_loader):
    val_loss = n_correct = n_word = 0

    st = time.time()
    for i, data in enumerate(val_loader):
        with torch.no_grad():
            data = [item.cuda() for item in data]

        instr_ids, instr_labels, instr_mask, pad_traj_views, traj_mask = data
        scores = _model('mlm',
                        instr_ids=instr_ids,
                        instr_labels=instr_labels,
                        instr_mask=instr_mask,
                        image_feat=pad_traj_views,
                        image_mask=traj_mask)

        instr_labels = instr_labels[instr_labels != -1]
        loss = F.cross_entropy(scores, instr_labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == instr_labels).sum().item()
        n_word += instr_labels.numel()

    tot_time = time.time() - st
    val_loss /= n_word
    acc = n_correct / n_word

    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_word / tot_time}

    LOGGER.info(f"Finished mlm validation in {int(tot_time)} seconds, acc is: {acc * 100:.2f}")
    return val_log


@torch.no_grad()
def validate_nap(_model, val_loader):
    val_loss = n_correct = n_data = 0

    st = time.time()
    for i, data in enumerate(val_loader):
        with torch.no_grad():
            data = [item.cuda() for item in data]

        instr_ids, instr_mask, candidate_views, candidate_mask, teacher_action = data
        scores = _model('nap',
                        instr_ids=instr_ids,
                        instr_mask=instr_mask,
                        image_feat=candidate_views,
                        image_mask=candidate_mask,
                        teacher_action=teacher_action)

        loss = F.cross_entropy(scores, teacher_action, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == teacher_action).sum().item()
        n_data += teacher_action.numel()

    tot_time = time.time() - st
    val_loss /= n_data
    acc = n_correct / n_data

    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_data / tot_time}

    LOGGER.info(f"Finished nap validation in {int(tot_time)} seconds, acc is: {acc * 100:.2f}")
    return val_log


@torch.no_grad()
def validate_nar(_model, val_loader):
    val_loss = n_correct = n_data = 0

    st = time.time()
    for i, data in enumerate(val_loader):
        with torch.no_grad():
            data = [item.cuda() for item in data]

        instr_ids, instr_mask, candidate_views, candidate_mask, teacher_action = data
        scores = _model('nar',
                        instr_ids=instr_ids,
                        instr_mask=instr_mask,
                        image_feat=candidate_views,
                        image_mask=candidate_mask,
                        teacher_action=teacher_action)

        loss = F.mse_loss(scores, teacher_action, reduction='sum')
        val_loss += loss.item()
        n_data += scores.size(0)

    tot_time = time.time() - st
    val_loss /= n_data

    val_log = {'loss': val_loss,
               'tok_per_s': n_data / tot_time}

    LOGGER.info(f"Finished nap validation in {int(tot_time)} seconds, val_loss is: {val_loss * 100:.2f}")
    return val_log


@torch.no_grad()
def validate_tom(_model, val_loader):
    val_loss = n_correct = n_data = 0

    st = time.time()
    for i, data in enumerate(val_loader):
        with torch.no_grad():
            data = [item.cuda() for item in data]

        instr_ids, instr_mask, pad_traj_views, pad_traj_index, traj_mask, traj_labels = data

        scores = _model('tom',
                        instr_ids=instr_ids,
                        instr_mask=instr_mask,
                        image_feat=(pad_traj_views, pad_traj_index),
                        image_mask=traj_mask,
                        teacher_action=traj_labels)

        loss = F.cross_entropy(scores, traj_labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == traj_labels).sum().item()
        n_data += traj_labels.numel()

    tot_time = time.time() - st
    val_loss /= n_data
    acc = n_correct / n_data

    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_data / tot_time}

    LOGGER.info(f"Finished tom validation in {int(tot_time)} seconds, acc is: {acc * 100:.2f}")
    return val_log


@torch.no_grad()
def validate_itm(_model, val_loader):
    val_loss = n_correct = n_data = 0

    st = time.time()
    for i, data in enumerate(val_loader):
        with torch.no_grad():
            data = [item.cuda() for item in data]

        instr_ids, instr_mask, pad_traj_views, traj_mask, traj_labels = data

        scores = _model('itm',
                        instr_ids=instr_ids,
                        instr_mask=instr_mask,
                        image_feat=pad_traj_views,
                        image_mask=traj_mask,
                        teacher_action=traj_labels)

        loss = F.cross_entropy(scores, traj_labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == traj_labels).sum().item()
        n_data += traj_labels.numel()

    tot_time = time.time() - st
    val_loss /= n_data
    acc = n_correct / n_data

    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_data / tot_time}

    LOGGER.info(f"Finished itm validation in {int(tot_time)} seconds, acc is: {acc * 100:.2f}")
    return val_log
