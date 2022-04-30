import logging
import sys

from random import random as rd
import torch.nn as nn
from transformers import BertPreTrainedModel

from .vilmodel_bert_vit_lx import BertLayerNorm, BertOnlyMLMHead
from .vilmodel_bert_vit_lx import VlnModel

sys.path.append('..')
from utils.parameters import args

logger = logging.getLogger(__name__)


class NextCandidatePrediction(nn.Module):
    """
    implement on the candidate embedding, choose the next candidate viewpoint
    """

    def __init__(self, hidden_size, dropout_rate):
        super(NextCandidatePrediction, self).__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)


class NextCandidateRegression(nn.Module):
    """
    implement on the candidate embedding, choose the next candidate viewpoint
    """

    def __init__(self, hidden_size, dropout_rate):
        super(NextCandidateRegression, self).__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 2))

    def forward(self, x):
        return self.net(x)


class TrajOrderPrediction(nn.Module):
    """
    implement on the candidate embedding, choose the next candidate viewpoint
    """

    def __init__(self, hidden_size, dropout_rate):
        super(TrajOrderPrediction, self).__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 4))

    def forward(self, x):
        return self.net(x)


class InstruTrajPrediction(nn.Module):
    """
    implement on the [CLS] embedding, predict whether the instru and traj is matched pair
    """

    def __init__(self, hidden_size, dropout_rate):
        super(InstruTrajPrediction, self).__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 2))

    def forward(self, x):
        return self.net(x)


def _compute_masked_hidden(hidden, mask):
    """
    get only the masked region (don't compute unnecessary hiddens)
    """
    mask = mask.unsqueeze(-1).expand_as(hidden)
    hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
    return hidden_masked


class VlnModelPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # proxy head
        if 'mlm' in args.proxy:
            self.mlm_head = BertOnlyMLMHead(self.config)
        if 'nap' in args.proxy:
            self.next_action_pre = NextCandidatePrediction(self.config.hidden_size, self.config.pred_head_dropout_prob)
        if 'nar' in args.proxy:
            self.next_action_reg = NextCandidateRegression(self.config.hidden_size, self.config.pred_head_dropout_prob)
        if 'tom' in args.proxy:
            self.tom_head = TrajOrderPrediction(self.config.hidden_size, self.config.pred_head_dropout_prob)
        if 'itm' in args.proxy:
            self.itm_head = InstruTrajPrediction(self.config.hidden_size, self.config.pred_head_dropout_prob)

        # enviroument dropout
        self.drop_env = nn.Dropout(p=args.featdropout)

        # initial the weights excpet for the lxmert vlnmodel
        self.apply(self._init_weights)
        logger.info("Finish initializing the vlnpretrian head randomly!")

        if args.lxmert_pretrain:
            # use the pretrained lxmert weights to initial the vlnmodel
            self.bert = VlnModel.from_pretrained('unc-nlp/lxmert-base-uncased', config=config)
            logger.info("Finish initializing the lxmert vlnmodel with the pretrained weights!")
        else:
            self.bert = VlnModel(config=config)
            logger.info("Finish initializing the lxmert vlnmodel randomly!")

        # tie the mlm decoder with the bert embedding weights
        self.tie_weights()

    def tie_weights(self):
        if 'mlm' in args.proxy:
            self._tie_or_clone_weights(self.mlm_head.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)

    def forward(self,
                task,
                instr_ids=None,
                instr_labels=None,
                instr_mask=None,
                image_feat=None,
                image_mask=None,
                teacher_action=None):
        probs = rd()
        if probs < 0.3:
            if 'tom' == task:
                image_feat[0][..., :-args.angle_feat_dim] = self.drop_env(image_feat[0][..., :-args.angle_feat_dim])
            else:
                image_feat[..., :-args.angle_feat_dim] = self.drop_env(image_feat[..., :-args.angle_feat_dim])

        if 'mlm' == task:
            pooled_output, lang_output, visual_output = self.bert(input_ids=instr_ids,
                                                                  attention_mask=instr_mask,
                                                                  visual_feats=image_feat,
                                                                  visual_attention_mask=image_mask)

            # only compute masked tokens for better efficiency
            masked_output = _compute_masked_hidden(lang_output, instr_labels != -1)
            prediction_scores = self.mlm_head(masked_output)

            return prediction_scores

        elif 'nap' == task:
            pooled_output, lang_output, visual_output = self.bert(input_ids=instr_ids,
                                                                  attention_mask=instr_mask,
                                                                  visual_feats=image_feat,
                                                                  visual_attention_mask=image_mask)

            # combine text and visual to predict next action
            prediction_scores = self.next_action_pre(visual_output * lang_output[:, 0:1, :]).squeeze(-1)

            return prediction_scores

        elif 'nar' == task:
            pooled_output, lang_output, visual_output = self.bert(input_ids=instr_ids,
                                                                  attention_mask=instr_mask,
                                                                  visual_feats=image_feat,
                                                                  visual_attention_mask=image_mask)

            # combine text and visual to predict next action
            prediction_scores = self.next_action_reg(lang_output[:, 0, :])

            return prediction_scores

        elif 'tom' == task:
            pad_traj_views, pad_traj_index = image_feat

            pooled_output, lang_output, visual_output = self.bert(input_ids=instr_ids,
                                                                  attention_mask=instr_mask,
                                                                  visual_feats=pad_traj_views,
                                                                  visual_attention_mask=image_mask)

            # only compute masked tokens for better efficiency
            # masked_output = _compute_masked_hidden(visual_output, pad_traj_index)
            masked_output = visual_output[pad_traj_index].contiguous().view(visual_output.shape[0], -1, visual_output.shape[-1])
            prediction_scores = self.tom_head(masked_output)

            return prediction_scores

        elif 'itm' == task:
            pooled_output, lang_output, visual_output = self.bert(input_ids=instr_ids,
                                                                  attention_mask=instr_mask,
                                                                  visual_feats=image_feat,
                                                                  visual_attention_mask=image_mask)

            # only compute masked tokens for better efficiency
            prediction_scores = self.itm_head(lang_output[:, 0, :])

            return prediction_scores

        else:
            raise ValueError('invalid task')
