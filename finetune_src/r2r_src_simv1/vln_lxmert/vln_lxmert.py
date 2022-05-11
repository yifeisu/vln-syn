# VLN-CLIP, 2022, by yifeisu
import sys
import torch
import torch.nn as nn

sys.path.append('..')

from param import args
from vln_lxmert.vln_lxmert_init import get_vlnlxmert_models


class NextCandidatePrediction(nn.Module):
    """
    implement on the candidate embedding, choose the next candidate viewpoint
    """

    def __init__(self, hidden_size, dropout_rate):
        super(NextCandidatePrediction, self).__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.LayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)


class VLNLXMERT(nn.Module):
    def __init__(self, feature_size=512 + 128):
        super(VLNLXMERT, self).__init__()
        print('\n Initalizing the VLNLXMERT model ...')
        self.feature_size = feature_size

        # initialize the VLNClip
        self.vln_lxmert = get_vlnlxmert_models(args, config=None)

        hidden_size = self.vln_lxmert.config.hidden_size
        layer_norm_eps = self.vln_lxmert.config.layer_norm_eps

        # enviroument dropout
        self.drop_env = nn.Dropout(p=args.featdropout)

        # projection: state + action -> state hidden_size
        self.state_action_project = nn.Sequential(nn.Linear(hidden_size + args.angle_feat_size, hidden_size, bias=True))
        self.state_action_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # projection: lang + visual -> single hidden_size
        self.lang_vis_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.state_prj = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.state_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # hame-like decision making.
        if args.decision_mode == 'hamt':
            # combine text and visual to predict next action
            self.next_action_pre = NextCandidatePrediction(hidden_size, 0.3)
            print("Using simlator version 1, and hamt-like decision making.")
        elif args.decision_mode == 'recbert':
            print("Using simlator version 1, and recbert-like decision making.")
        else:
            raise ValueError('unknown decision_mode.')

    def forward(self,
                mode,
                lang_feats,
                lang_attention_mask=None,
                visual_feats=None,
                visual_attention_mask=None,
                action_angle_feat=None):

        if mode == 'language':
            pooler_output, encoded_sentence = self.vln_lxmert(mode,
                                                              input_ids=lang_feats,
                                                              attention_mask=lang_attention_mask)

            return pooler_output, encoded_sentence

        elif mode == 'visual':
            # update the state， concatenate the action feature and do the projection
            state_action = torch.cat([lang_feats[:, 0, :], action_angle_feat], dim=1)
            # do projection
            state_action = self.state_action_project(state_action)
            state_action = self.state_action_ln(state_action)

            # finish updating
            if args.update_state:
                state_lang = torch.cat([state_action.unsqueeze(1), lang_feats[:, 1:, :]], dim=1)
            else:
                state_lang = lang_feats

            # drop the env vision features
            visual_feats[..., :-args.angle_feat_size] = self.drop_env(visual_feats[..., :-args.angle_feat_size])

            h_t, logits, attended_language, attended_visual, lang_output, visual_output = self.vln_lxmert(mode,
                                                                                                          input_ids=state_lang,  # lang_feats
                                                                                                          attention_mask=lang_attention_mask,
                                                                                                          visual_feats=visual_feats,
                                                                                                          visual_attention_mask=visual_attention_mask)

            vis_lang = self.lang_vis_ln(attended_language * attended_visual)
            state_l_v = torch.cat([h_t, vis_lang], dim=-1)
            state = self.state_prj(state_l_v)
            state = self.state_ln(state)

            if args.decision_mode == 'recbert':
                preds = logits
            elif args.decision_mode == 'hamt':
                # combine text and visual to predict next action
                preds = self.next_action_pre(visual_output * lang_output[:, 0:1, :]).squeeze(-1)
            else:
                raise ValueError('unknown decision_mode.')

            return state, preds

        else:
            ModuleNotFoundError


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()
