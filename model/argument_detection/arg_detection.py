import torch
import torch.nn as nn
from transformers import DistilBertModel
from transformers import BertModel
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np


def lengths_to_mask(lengths, max_len=None, dtype=None):
    """
    Converts a "lengths" tensor to its binary mask representation.

    Based on: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397

    :lengths: N-dimensional tensor
    :returns: N*max_len dimensional tensor. If max_len==None, max_len=max(lengtsh)
    """
    assert len(lengths.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or lengths.max().item()
    mask = torch.arange(
        max_len,
        device=lengths.device,
        dtype=lengths.dtype).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=lengths.device)
    return mask


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """
    Masked verstion of the 1D Batch normalization.

    Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py

    Receives a N-dim tensor of sequence lengths per batch element
    along with the regular input for masking.

    Check pytorch's BatchNorm1d implementation for argument details.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats
        )

    def forward(self, inp, mask):
        self._check_input_dim(inp)

        exponential_average_factor = 0.0

        n = mask.sum()
        mask = mask / n.float()

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training and n > 1:
            # Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
            # variance, we do not need to make any tensor shape manipulation.
            # mean = E[X] is simply the sum-product of our "probability" mask with the input...
            mean = (mask * inp).sum([0, 2])
            # ...whereas Var(X) is directly derived from the above formulae
            # This should be numerically equivalent to the biased sample variance
            var = (mask * inp ** 2).sum([0, 2]) - mean ** 2
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # Update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        inp = (inp - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        if self.affine:
            inp = inp * self.weight[None, :, None] + self.bias[None, :, None]

        return inp


class ModelRichContext(nn.Module):
    def __init__(self, config):
        super(ModelRichContext, self).__init__()
        self.config = config
        self.embedding_dim = config.EMBEDDING_DIM
        self.dropout = config.dropout
        self.entity_type = len(self.config.fact_container.entity_to_ids)
        self.arg_roles = config.arg_roles

        self.pretrained_weights = str(config.pretrained_weights)
        if self.pretrained_weights == 'distilbert-base-uncased':
            self.bert = DistilBertModel.from_pretrained(self.pretrained_weights)
        elif self.pretrained_weights == 'bert':
            self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True, output_hidden_states=True)
        elif self.pretrained_weights == 'spanbert':
            self.bert = AutoModel.from_pretrained("SpanBERT/spanbert-large-cased")
        elif self.pretrained_weights == 'bertlarge':
            self.bert = BertModel.from_pretrained('bert-large-uncased', output_attentions=True, output_hidden_states=True)
        else:
            self.bert = None

        self.extra_bert = config.extra_bert
        self.use_extra_bert = config.use_extra_bert
        if self.use_extra_bert:
            self.embedding_dim *= 2

        self.n_hid = 768
        self.linear = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim*6, self.n_hid),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_hid, 1)
        )
        self.sqrt_d = np.sqrt(self.embedding_dim)

    def get_fist_subword_embeddings(self, all_embeddings, idxs_to_collect, bert_sentence_lengths):
        """
        Pick first subword embeddings with the indices list idxs_to_collect
        :param all_embeddings:
        :param idxs_to_collect:
        :param targets:
        :param bert_sentence_lengths:
        :return:
        """
        sent_embeddings = []
        N = all_embeddings.shape[0]  # it's equivalent to N=len(all_embeddings)

        # Other two mode need to be taken care of the issue
        # that the last index becomes the [SEP] after argument types
        arg_type_embeddings = []
        bert_sentence_lengths = bert_sentence_lengths.long()
        for i in range(N):
            this_idxs_to_collect = idxs_to_collect[i]
            this_idxs_to_collect = this_idxs_to_collect[this_idxs_to_collect>0]
            collected = all_embeddings[i, this_idxs_to_collect[:-2]]  # collecting a slice of tensor
            sent_embeddings.append(collected)
            second_last_sep_index = this_idxs_to_collect[-2]

            # argument type embedding
            arg_type_embedding = all_embeddings[i, second_last_sep_index+1:bert_sentence_lengths[i]]
            arg_type_embeddings.append(arg_type_embedding)

        max_sent_len = idxs_to_collect.shape[1] - 2
        max_len_arg = 9

        for i in range(N):
            arg_type_embeddings[i] = torch.cat((arg_type_embeddings[i], torch.zeros(max_len_arg - len(arg_type_embeddings[i]), self.embedding_dim).cuda()))
            sent_embeddings[i] = torch.cat((sent_embeddings[i], torch.zeros(max_sent_len - len(sent_embeddings[i]), self.embedding_dim).cuda()))

        sent_embeddings = torch.stack(sent_embeddings)
        arg_type_embeddings = torch.stack(arg_type_embeddings)
        return sent_embeddings, arg_type_embeddings

    @staticmethod
    def get_trigger_embeddings(sent_embeddings, is_triggers):
        """
        Select trigger embedding with the is_trigger mask
        :param sent_embeddings:
        :param is_triggers:
        :return:
        """
        return torch.sum(sent_embeddings*is_triggers.unsqueeze(-1)/torch.sum(is_triggers, dim=1).unsqueeze(-1).unsqueeze(-1), dim=1)

    def generate_concat(self, sent_embeddings, trigger_embeddings):
        trigger_count = trigger_embeddings.shape[1]
        sent_len = sent_embeddings.shape[1]

        trigger_embeddings = torch.unsqueeze(trigger_embeddings, 2).repeat(1,1,sent_len, 1)
        sent_embeddings = torch.unsqueeze(sent_embeddings, 1).repeat(1, trigger_count, 1, 1)
        if self.config.without_trigger:
            sent_trigger_cat = torch.cat((sent_embeddings, trigger_embeddings), -1)
            return sent_trigger_cat
        else:
            return sent_embeddings

    def forward(self, sentence_batch, idxs_to_collect, is_triggers, bert_sentence_lengths,
                arg_weight_matrices, arg_mapping, entity_mapping):
        # get embeddings
        sent_mask = (sentence_batch != 0) * 1
        all_embeddings, _, hidden_states, hidden_layer_att = self.bert(sentence_batch.long(), attention_mask=sent_mask)

        if self.use_extra_bert:
            extra_bert_outputs = hidden_states[self.extra_bert]
            all_embeddings = torch.cat([all_embeddings, extra_bert_outputs], dim=2)

        sent_embeddings, arg_embeddings = self.get_fist_subword_embeddings(all_embeddings, idxs_to_collect, bert_sentence_lengths)
        entity_embeddings = sent_embeddings.permute(0, 2, 1).matmul( entity_mapping).permute(0,2,1)
        trigger_candidates = self.get_trigger_embeddings(sent_embeddings, is_triggers)

        arg_embeddings = arg_embeddings.transpose(1,2).matmul(arg_weight_matrices.float()).transpose(1, 2)
        _trigger = trigger_candidates.unsqueeze(1).repeat(1, entity_embeddings.shape[1], 1)

        # token to argument attention
        token2arg_score = torch.sum(entity_embeddings.unsqueeze(2) * arg_embeddings.unsqueeze(1), dim=-1) * (1 / self.sqrt_d)

        # attention weights
        token2arg_softmax = ((token2arg_score-5)/2).unsqueeze(-1)
        arg2token_softmax = ((token2arg_score-5)/2).unsqueeze(-1)

        token_argAwared = torch.sum(arg_embeddings.unsqueeze(1) * token2arg_softmax, dim=2)   # b * sent_len * 768
        arg_tokenAwared = torch.sum(entity_embeddings.unsqueeze(2) * arg2token_softmax, dim=1)  # b *  arg_len * 768

        # bidirectional attention
        A_h2u = token_argAwared.unsqueeze(2).repeat(1,1,arg_embeddings.shape[1],1)
        A_u2h = arg_tokenAwared.unsqueeze(1).repeat(1,entity_embeddings.shape[1],1,1)
        # argumentation embedding
        U_ = arg_embeddings.unsqueeze(1).repeat(1,entity_embeddings.shape[1],1,1)

        # entity-entity attention
        last0_layer_atten = self.select_hidden_att(hidden_layer_att[-1], idxs_to_collect)
        last1_layer_atten = self.select_hidden_att(hidden_layer_att[-2], idxs_to_collect)
        last2_layer_atten = self.select_hidden_att(hidden_layer_att[-3], idxs_to_collect)
        token2token_softmax = (last0_layer_atten + last1_layer_atten + last2_layer_atten)/3
        A_h2h = token2token_softmax.matmul(sent_embeddings).unsqueeze(2).repeat(1, 1, arg_embeddings.shape[1], 1)
        H_ = sent_embeddings.unsqueeze(2).repeat(1, 1, arg_embeddings.shape[1],1)
        A_h2h = A_h2h.permute(0, 2, 3, 1).matmul(entity_mapping.unsqueeze(1)).permute(0, 3, 1, 2)
        H_ = H_.permute(0, 2, 3, 1).matmul(entity_mapping.unsqueeze(1)).permute(0, 3, 1, 2)

        # arg role to arg role attention
        arg2arg_softmax = F.softmax(arg_embeddings.matmul(arg_embeddings.transpose(-1,-2)), dim=-1)
        A_u2u = arg2arg_softmax.matmul(arg_embeddings).unsqueeze(1).repeat(1, entity_embeddings.shape[1], 1, 1)

        latent = torch.cat((H_, U_, A_h2u, A_h2h, A_u2h, A_u2u), dim=-1)
        score = self.linear(latent)
        score = self.map_arg_to_ids(score, arg_mapping)
        return score

    def map_arg_to_ids(self, score, arg_mapping):
        """
        Here we put each argument embedding back to its original place.
        In the input [CLS] sentence [SEP] arguments [SEP],
        arguments contains arguments of the specific trigger type.
        Thus we need to put them back to their actual indices
        :param score:
        :param arg_mapping:
        :return:
        """
        b, s, _ = score.shape
        d = self.arg_roles+1
        new_score = -1e6 * torch.ones(b, s, d).cuda()
        for i in range(b):
            ids = arg_mapping[i][arg_mapping[i] < self.arg_roles+1]
            new_score[i, :, ids] = score[i, :, :len(ids)]
        return new_score

    @staticmethod
    def select_hidden_att(hidden_att, ids_to_collect):
        """
        Pick attentions from hidden layers
        :param hidden_att: of dimension (batch_size, embed_length, embed_length)
        :return:
        """
        N = hidden_att.shape[0]
        sent_len = ids_to_collect.shape[1] - 2
        hidden_att = torch.mean(hidden_att, 1)
        hidden_att_selected = torch.zeros(N, sent_len, sent_len).cuda()

        for i in range(N):
            to_collect = ids_to_collect[i]
            to_collect = to_collect[to_collect>0][:-2]
            collected = hidden_att[i, to_collect][:,to_collect]  # collecting a slice of tensor
            hidden_att_selected[i, :len(to_collect), :len(to_collect)] = collected

        return hidden_att_selected/(torch.sum(hidden_att_selected, dim=-1, keepdim=True)+1e-9)