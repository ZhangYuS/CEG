import torch
import torch.nn as nn
from scripts.evaluation.classification_evaluation import evaluate as classfication_evaluate


class GateModel(nn.Module):

    def __init__(self, hparams, n_embd):
        super().__init__()
        self.hparams = hparams
        self.n_embd = n_embd
        self.gate_mode = hparams.gate_mode
        self.select_loss = hparams.select_loss
        self.select_mode = hparams.select_mode

        if self.hparams.use_gate:
            if self.hparams.gate_mode == 'double':
                # gate = Wx + Uy + b
                self.W = nn.Linear(self.n_embd, 1, bias=False)
                self.U = nn.Linear(self.n_embd, 1)
            elif self.hparams.gate_mode == 'single':
                self.gate = nn.Bilinear(self.n_embd, self.n_embd, 1)
            else:
                raise ValueError('parameter gate_mode error')

    def forward(self, story_hidden_state, inference_hidden_state, score_soft_label, score_hard_label):
        batch_size = story_hidden_state.shape[0]
        story_board_hidden_state = story_hidden_state[:, None, :].repeat(1, self.hparams.max_triple_num, 1)
        if self.hparams.gate_mode == 'double':
            sentence_logits = self.W(story_board_hidden_state) + self.U(inference_hidden_state)
        elif self.hparams.gate_mode == 'single':
            sentence_logits = self.gate(story_board_hidden_state, inference_hidden_state)
        else:
            raise ValueError('parameter gate_mode error')
        sentence_score = nn.Sigmoid()(sentence_logits).squeeze(dim=-1)

        if self.hparams.select_loss == 'hard':
            select_loss = nn.BCELoss()(sentence_score, score_hard_label.float())
        elif self.hparams.select_loss == 'soft':
            select_loss = nn.BCELoss()(sentence_score, score_soft_label.float())
        elif self.hparams.select_loss == 'both':
            select_loss = nn.BCELoss()(sentence_score, score_soft_label.float()) + nn.BCELoss()(
                sentence_score, score_hard_label.float())
        else:
            raise ValueError('parameter select_loss error')


        if self.hparams.filter_mode == 'topk':
            threshold = torch.kthvalue(sentence_score, k=self.hparams.max_triple_num - self.hparams.select_k, dim=1,
                                       keepdim=True)[0]
            select_index = (sentence_score > threshold).long()
        elif self.hparams.filter_mode == 'threshold':
            select_index = (sentence_score > self.hparams.select_threshold).long()
        else:
            raise ValueError(f'parameter select_mode error')

        ret_classification = None
        if self.hparams.filter_mode == 'topk' or self.hparams.filter_mode == 'threshold':
            ret_list = []
            ret_classification = {}
            for i in range(batch_size):
                ret_list.append(classfication_evaluate(select_index[i].tolist(), score_hard_label[i].tolist()))
            for k in ret_list[0].keys():
                ret_classification[k] = sum([x[k] for x in ret_list]) / batch_size

        return sentence_score, select_index, select_loss, ret_classification
