import time
import random
import csv

import base64
import logging
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

csv.field_size_limit(2500000)
logger = logging.getLogger(__name__)


def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1) > (torch.LongTensor(length) - 1).unsqueeze(1))
    return mask


def read_img_features(feature_path, views):
    logger.info("Start loading the image feature from %s (~15 seconds)" % feature_path)
    start = time.time()

    tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
    features = {}
    with open(feature_path, "r") as tsv_in_file:  # Open the tsv file.
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
        for item in reader:
            long_id = item['scanId'] + "_" + item['viewpointId']
            # Feature of long_id is (36, 2048)
            features[long_id] = np.frombuffer(base64.decodebytes(item['features'].encode('ascii')),
                                              dtype=np.float32).reshape((views, -1))

    logger.info("Finish Loading the image feature in %0.4f seconds" % (time.time() - start))
    return features


def angle_feature(heading, elevation):
    return np.array([np.sin(heading), np.cos(heading), np.sin(elevation), np.cos(elevation)] * (128 // 4),
                    dtype=np.float32)


def random_word(tokens_ids, vocab_range, mask_token_id, pad_token_id):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens_ids: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :param mask_token_id: int, id of 'mask' token
    :param pad_token_id:

    :return: (list of int, list of int), masked tokens and related labels for LM prediction
    """
    output_tokens, output_label = list(), list()
    for i, token in enumerate(tokens_ids):
        if token == pad_token_id:
            output_tokens.append(token)
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
        else:
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15
                # 80%, randomly change token to mask token
                if prob < 0.8:
                    output_tokens.append(mask_token_id)
                # 10% randomly change token to random token
                elif prob < 0.9:
                    output_tokens.append(random.randint(*vocab_range))
                # rest 10% keep current token
                else:
                    output_tokens.append(token)

                # append current token to output (we will predict these later)
                output_label.append(token)
            else:
                output_tokens.append(token)
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[1] = tokens_ids[1]
        output_tokens[1] = mask_token_id

    return output_tokens, output_label


# -------------------------------------------------------------------------------------- #
# create the dataset and corresponding collate function
# -------------------------------------------------------------------------------------- #
class MlmDataset(Dataset):
    def __init__(self, json_data, features):
        """
        json_path:
        features:
        """
        super(MlmDataset, self).__init__()

        self.image_feat = features
        self.data = json_data

        self.tok = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_range = [1996, 29611]  # manually checked in bert-base-uncased
        self.cls_token_id = self.tok.cls_token_id  # 101
        self.sep_token_id = self.tok.sep_token_id  # 102
        self.mask_token_id = self.tok.mask_token_id  # 103
        self.pad_token_id = self.tok.pad_token_id  # 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index].copy()
        # 1. mask the instruction
        instr = item['instr_ids']
        instr_ids, instr_labels = random_word(instr, self.vocab_range, self.mask_token_id, self.tok.pad_token_id)
        instr_ids, instr_labels = torch.LongTensor(instr_ids), torch.LongTensor(instr_labels)
        instr_mask = (instr_ids != self.pad_token_id).long()

        # 2. collect the trajectory views
        traj_views = list()
        for viewpoint in item["path"]:
            _long_id = '%s_%s' % (viewpoint[0], viewpoint[1])
            image_feat = self.image_feat[_long_id][viewpoint[2]]
            angle_feat = angle_feature(0, 0)
            traj_views.append(np.concatenate([image_feat, angle_feat], axis=0))
        traj_views = torch.from_numpy(np.vstack(traj_views))  # traj_len, feature_dim

        return instr_ids, instr_labels, instr_mask, traj_views


def mlm_collate(inputs):
    """
    :param inputs: a list of tuple sampled from dataset, eahc tuple is (instr_ids, instr_labels, instr_mask, traj_views)
    """
    instr_ids, instr_labels, instr_mask, traj_views = list(zip(*inputs))
    instr_ids, instr_labels, instr_mask = torch.vstack(instr_ids), torch.vstack(instr_labels), torch.vstack(instr_mask)

    # prepare the traj_views and mask, to same length
    traj_length = [traj_view.shape[0] for traj_view in traj_views]
    max_len = max(traj_length)
    pad_traj_views = torch.zeros([len(traj_views), max_len, traj_views[0].shape[1]], dtype=torch.float32)
    for i in range(len(traj_views)):
        for j in range(traj_length[i]):
            pad_traj_views[i, j, :] = traj_views[i][j]

    traj_mask = (length2mask(traj_length) == 0).long()

    return instr_ids, instr_labels, instr_mask, pad_traj_views, traj_mask


class NapDataset(Dataset):
    def __init__(self, json_data, features):
        super(NapDataset, self).__init__()

        self.image_feat = features
        self.pad_token_id = 0

        # split the each viewpoint
        self.viewpoint_data = list()
        for item in json_data:
            for index, point in enumerate(item['path']):
                new_item = dict()
                '''
                new_item = {
                'long_id': str,
                "instr_ids": [int...], 
                'cand_view_idex': [int...],
                'cand_rela_angle': [[float, float]...],
                'next_viewpointid': int}
                '''

                new_item['long_id'] = '%s_%s' % (point[0], point[1])
                new_item['instr_ids'] = item['instr_ids']
                new_item['cand_view_idex'] = item['cands_view_index'][index]
                new_item['cand_rela_angle'] = item['cands_rela_angle'][index]
                new_item['next_viewpointid'] = item['next_viewpointids'][index]
                self.viewpoint_data.append(new_item)

    def __len__(self):
        return len(self.viewpoint_data)

    def __getitem__(self, index):
        item = self.viewpoint_data[index].copy()
        # 1. prepare the instruction
        instr = item['instr_ids']
        instr_ids = torch.LongTensor(instr)
        instr_mask = (instr_ids != self.pad_token_id).long()

        # 2.prepare the candidate views of each point
        candidate_views = list()
        _long_id = item['long_id']
        for index, candidate in enumerate(item["cand_view_idex"]):
            image_feat = self.image_feat[_long_id][candidate]
            angle_feat = angle_feature(*item["cand_rela_angle"][index])
            candidate_views.append(np.concatenate([image_feat, angle_feat], axis=0))

        # 3.pad the stop 'views'
        pad_stop_cand = np.concatenate([np.zeros_like(image_feat, dtype=np.float32), angle_feature(0, 0)], axis=0)
        candidate_views.append(pad_stop_cand)

        candidate_views = torch.from_numpy(np.vstack(candidate_views))

        # 4.prepare the label
        if item['next_viewpointid'] != -1:
            teacher_action = torch.tensor([item['next_viewpointid']])
        else:
            teacher_action = torch.tensor([candidate_views.shape[0]-1])

        return instr_ids, instr_mask, candidate_views, teacher_action


def nap_collate(inputs):
    """
    :param inputs: a list of tuple, [(instr_ids, instr_mask, candidate_views, teacher_action)], ...]
    """
    instr_ids, instr_mask, candidate_views, teacher_action = list(zip(*inputs))
    instr_ids, instr_mask, teacher_action = torch.vstack(instr_ids), torch.vstack(instr_mask), torch.cat(teacher_action)

    # prepare the candidate views and mask, to same length
    # candidate_views, a tuple of tensors, ( (candidate_len, feat_dim),... )
    cand_leng = [candidate.shape[0] for candidate in candidate_views]
    max_leng = max(cand_leng)
    pad_candidate_views = torch.zeros([len(candidate_views), max_leng, candidate_views[0].shape[1]])
    for i in range(len(candidate_views)):
        for j in range(cand_leng[i]):
            pad_candidate_views[i, j, :] = candidate_views[i][j]

    candidate_mask = (length2mask(cand_leng) == 0).long()
    return instr_ids, instr_mask, pad_candidate_views, candidate_mask, teacher_action


class NarDataset(Dataset):
    def __init__(self, json_data, features):
        super(NarDataset, self).__init__()

        self.image_feat = features
        self.pad_token_id = 0

        # split the each viewpoint
        self.viewpoint_data = list()
        for item in json_data:
            for index, point in enumerate(item['path'][:-1]):
                new_item = dict()
                '''
                new_item = {
                'long_id': str,
                "instr_ids": [int...], 
                'cand_view_idex': [int...],
                'cand_rela_angle': [[float, float]...],
                'next_viewpointid': int}
                '''

                new_item['long_id'] = '%s_%s' % (point[0], point[1])
                new_item['instr_ids'] = item['instr_ids']
                new_item['cand_view_idex'] = item['cands_view_index'][index]
                new_item['cand_rela_angle'] = item['cands_rela_angle'][index]
                next_viewpointid = item['next_viewpointids'][index]
                if next_viewpointid != -1:
                    new_item['teacher_action'] = new_item['cand_rela_angle'][next_viewpointid]
                else:
                    new_item['teacher_action'] = [0, 0]

                self.viewpoint_data.append(new_item)

    def __len__(self):
        return len(self.viewpoint_data)

    def __getitem__(self, index):
        item = self.viewpoint_data[index].copy()
        # 1. prepare the instruction
        instr = item['instr_ids']
        instr_ids = torch.LongTensor(instr)
        instr_mask = (instr_ids != self.pad_token_id).long()

        # 2.prepare the candidate views of the random selected viewpoint in path
        candidate_views = list()
        _long_id = item['long_id']
        for index, candidate in enumerate(item["cand_view_idex"]):
            image_feat = self.image_feat[_long_id][candidate]
            angle_feat = angle_feature(*item["cand_rela_angle"][index])
            candidate_views.append(np.concatenate([image_feat, angle_feat], axis=0))

        # 3.pad the stop 'views'
        pad_stop_cand = np.concatenate([np.zeros_like(image_feat, dtype=np.float32), angle_feature(0, 0)], axis=0)
        candidate_views.append(pad_stop_cand)

        candidate_views = torch.from_numpy(np.vstack(candidate_views))

        # 4.prepare the label
        teacher_action = torch.tensor(item['teacher_action'])
        return instr_ids, instr_mask, candidate_views, teacher_action


def nar_collate(inputs):
    """
    :param inputs: a list of tuple, [(instr_ids, instr_mask, candidate_views, teacher_action)], ...]
    """
    instr_ids, instr_mask, candidate_views, teacher_action = list(zip(*inputs))
    instr_ids, instr_mask, teacher_action = torch.vstack(instr_ids), torch.vstack(instr_mask), torch.vstack(teacher_action)

    # prepare the candidate views and mask, to same length
    # candidate_views, a tuple of tensors, ( (candidate_len, feat_dim),... )
    cand_leng = [candidate.shape[0] for candidate in candidate_views]
    max_leng = max(cand_leng)
    pad_candidate_views = torch.zeros([len(candidate_views), max_leng, candidate_views[0].shape[1]])
    for i in range(len(candidate_views)):
        for j in range(cand_leng[i]):
            pad_candidate_views[i, j, :] = candidate_views[i][j]

    candidate_mask = (length2mask(cand_leng) == 0).long()
    return instr_ids, instr_mask, pad_candidate_views, candidate_mask, teacher_action


class TomDataset(Dataset):
    def __init__(self, json_data, features):
        """
        json_path:
        features:
        """
        super(TomDataset, self).__init__()

        self.image_feat = features
        self.data = json_data

        self.pad_token_id = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index].copy()
        # 1. mask the instruction
        instr = item['instr_ids']
        instr_ids = torch.LongTensor(instr)
        instr_mask = (instr_ids != self.pad_token_id).long()

        # 2. collect the trajectory views
        traj_views = list()
        for viewpoint in item["path"]:
            _long_id = '%s_%s' % (viewpoint[0], viewpoint[1])
            image_feat = self.image_feat[_long_id][viewpoint[2]]
            angle_feat = angle_feature(0, 0)
            traj_views.append(np.concatenate([image_feat, angle_feat], axis=0))
        traj_views = np.vstack(traj_views)  # traj_len x feature_dim

        # 3.randomly shuffle the trajetory order and return the label
        np_index_origin = np.random.choice(np.arange(traj_views.shape[0]), 4, replace=False)
        np_index_origin = np.sort(np_index_origin)
        np_index_shuffle = np_index_origin.copy()
        np.random.shuffle(np_index_shuffle)
        traj_views[np_index_origin] = traj_views[np_index_shuffle]

        # 4. np_index_shuffle
        np_index_labels = np.array([np.where(np_index_origin == i)[0][0] for i in np_index_shuffle])

        # 5.convert to torch.tensor
        traj_views, traj_labels = torch.from_numpy(traj_views), torch.from_numpy(np_index_labels).long()
        traj_index = torch.zeros([traj_views.shape[0]])
        for index in np_index_origin:
            traj_index[index] = 1

        return instr_ids, instr_mask, traj_views, traj_index, traj_labels


def tom_collate(inputs):
    instr_ids, instr_mask, traj_views, traj_index, traj_labels = list(zip(*inputs))
    instr_ids, instr_mask, traj_labels = torch.vstack(instr_ids), torch.vstack(instr_mask), torch.vstack(traj_labels)

    # prepare the traj_views and mask, to same length
    traj_length = [traj_view.shape[0] for traj_view in traj_views]
    max_len = max(traj_length)
    pad_traj_views = torch.zeros([len(traj_views), max_len, traj_views[0].shape[1]], dtype=torch.float32)
    pad_traj_index = torch.zeros([len(traj_views), max_len], dtype=torch.float32)
    for i in range(len(traj_views)):
        for j in range(traj_length[i]):
            pad_traj_views[i, j, :] = traj_views[i][j]
            pad_traj_index[i, j] = traj_index[i][j]

    traj_mask = (length2mask(traj_length) == 0).long()
    pad_traj_index = pad_traj_index.bool()

    return instr_ids, instr_mask, pad_traj_views, pad_traj_index, traj_mask, traj_labels


class ItmDataset(Dataset):
    def __init__(self, json_data, features):
        """
        json_path:
        features:
        """
        super(ItmDataset, self).__init__()

        self.image_feat = features
        self.data = json_data
        self.pad_token_id = 0

        self.meta_data = dict()
        for item in self.data:
            scan = item["traj_scan"]
            if self.meta_data.get(scan) is None:
                self.meta_data[scan] = [item["path"]]
            else:
                self.meta_data[scan].append(item["path"])
        self.scans = list(self.meta_data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index].copy()

        scan = item["traj_scan"]
        trajs = self.meta_data[scan].copy()  # instr_trajs: a list of tuple [(instr, traj)...]

        prob = random.random()
        if prob < 0.5:
            instr_raw = item["instr_ids"]
            trajs.remove(item["path"])
            traj_id = random.choice(range(len(trajs)))
            traj_raw = trajs[traj_id]
            itm_label = 1
        else:
            instr_raw, traj_raw = item["instr_ids"], item["path"]
            itm_label = 0

        # 1. mask the instruction
        instr_ids = torch.LongTensor(instr_raw)
        instr_mask = (instr_ids != self.pad_token_id).long()

        # 2. collect the trajectory views
        traj_views = list()
        for viewpoint in traj_raw:
            _long_id = '%s_%s' % (viewpoint[0], viewpoint[1])
            image_feat = self.image_feat[_long_id][viewpoint[2]]
            angle_feat = angle_feature(0, 0)
            traj_views.append(np.concatenate([image_feat, angle_feat], axis=0))
        traj_views = torch.from_numpy(np.vstack(traj_views))  # traj_len, feature_dim

        itm_label = torch.tensor([itm_label])

        return instr_ids, instr_mask, traj_views, itm_label


def itm_collate(inputs):
    instr_ids, instr_mask, traj_views, itm_label = list(zip(*inputs))
    instr_ids, instr_mask, traj_labels = torch.vstack(instr_ids), torch.vstack(instr_mask), torch.hstack(itm_label)

    # prepare the traj_views and mask, to same length
    traj_length = [traj_view.shape[0] for traj_view in traj_views]
    max_len = max(traj_length)
    pad_traj_views = torch.zeros([len(traj_views), max_len, traj_views[0].shape[1]], dtype=torch.float32)
    for i in range(len(traj_views)):
        for j in range(traj_length[i]):
            pad_traj_views[i, j, :] = traj_views[i][j]

    traj_mask = (length2mask(traj_length) == 0).long()

    return instr_ids, instr_mask, pad_traj_views, traj_mask, traj_labels
