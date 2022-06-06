import torch
import torch.nn as nn
from transformers import DistilBertModel
from transformers import BertModel
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np

class EntityDetection(nn.Module):
    def __init__(self, tag_to_ix, config, num_lstm_layers=2, bilstm=True):
        super(EntityDetection, self).__init__()
        self.embedding_dim = config.EMBEDDING_DIM
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.BOS_TAG_ID = self.tag_to_ix[config.START_TAG]
        self.EOS_TAG_ID = self.tag_to_ix[config.STOP_TAG]
        self.dropout_rate = 0.2
        self.config = config

        self.pretrained_weights = config.pretrained_weights
        if self.pretrained_weights == 'distilbert-base-uncased':
            self.bert = DistilBertModel.from_pretrained(self.pretrained_weights)
        elif self.pretrained_weights == 'bert':
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        elif self.pretrained_weights == 'bertlarge':
            self.bert = BertModel.from_pretrained('bert-large-uncased', output_attentions=True)
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.num_lstm_layers = num_lstm_layers
        self.bilstm = bilstm

        # Maps the output of the BERT directly into tag space.
        self.hidden2tag = nn.Linear(self.embedding_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *from* i *to* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # no transitions allowed to the beginning of sentence
        self.transitions.data[:, self.BOS_TAG_ID] = -10000.0
        # no transition alloed from the end of sentence
        self.transitions.data[self.EOS_TAG_ID, :] = -10000.0

    def _forward_alg(self, emissions, mask):
        """Compute the partition function in log-space using the forward-algorithm.
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        Returns:
            torch.Tensor: the partition scores for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length, nb_labels = emissions.shape

        # in the first iteration, BOS will have all the scores
        alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]

        for i in range(1, seq_length):
            alpha_t = []

            for tag in range(nb_labels):
                # get the emission for the current tag
                e_scores = emissions[:, i, tag]

                # broadcast emission to all labels
                # since it will be the same for all previous tags
                # (bs, nb_labels)
                e_scores = e_scores.unsqueeze(1)

                # transitions from something to our tag
                t_scores = self.transitions[:, tag]

                # broadcast the transition scores to all batches
                # (bs, nb_labels)
                t_scores = t_scores.unsqueeze(0)

                # combine current scores with previous alphas
                # since alphas are in log space (see logsumexp below),
                # we add them instead of multiplying
                scores = e_scores + t_scores + alphas

                # add the new alphas for the current tag
                alpha_t.append(torch.logsumexp(scores, dim=1))

            # create a torch matrix from alpha_t
            # (bs, nb_labels)
            new_alphas = torch.stack(alpha_t).t()

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas

        # add the scores for the final transition
        last_transition = self.transitions[:, self.EOS_TAG_ID]
        end_scores = alphas + last_transition.unsqueeze(0)

        # return a *log* of sums of exps
        return torch.logsumexp(end_scores, dim=1)

    def _score_sentence(self, emissions, tags, mask):
        """Compute the scores for a given batch of emissions with their tags.
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            tags (Torch.LongTensor): (batch_size, seq_len)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        Returns:
            torch.Tensor: Scores for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length = tags.shape
        scores = torch.zeros(batch_size).cuda()

        # save first and last tags to be used later
        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()

        # add the transition from BOS to the first tags for each batch
        t_scores = self.transitions[self.BOS_TAG_ID, first_tags]

        # add the [unary]    emission scores for the first tags for each batch
        # for all batches, the first word, see the correspondent emissions
        # for the first tags (which is a list of ids):
        # emissions[:, 0, [tag_1, tag_2, ..., tag_nblabels]]
        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()

        # the scores for a word is just the sum of both scores
        scores += e_scores + t_scores

        # now lets do this for each remaining word
        for i in range(1, seq_length):
            # we could: iterate over batches, check if we reached a mask symbol
            # and stop the iteration, but vecotrizing is faster due to gpu,
            # so instead we perform an element-wise multiplication
            is_valid = mask[:, i]

            previous_tags = tags[:, i - 1]
            current_tags = tags[:, i]

            # calculate emission and transition scores as we did before
            e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
            t_scores = self.transitions[previous_tags, current_tags]

            # apply the mask
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid

            scores += e_scores + t_scores

        # add the transition from the end tag to the EOS tag for each batch
        scores += self.transitions[last_tags, self.EOS_TAG_ID]

        return scores

    def viterbi_decode(self, emissions, mask):
        """Compute the viterbi algorithm to find the most probable sequence of labels
        given a sequence of emissions.
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        Returns:
            torch.Tensor: the viterbi score for the for each batch.
                Shape of (batch_size,)
            list of lists of ints: the best viterbi sequence of labels for each batch
        """
        batch_size, seq_length, nb_labels = emissions.shape
        alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]

        backpointers = []

        for i in range(1, seq_length):
            alpha_t = []
            backpointers_t = []

            for tag in range(nb_labels):
                # get the emission for the current tag and broadcast to all labels
                e_scores = emissions[:, i, tag]
                e_scores = e_scores.unsqueeze(1)

                # transitions from something to our tag and broadcast to all batches
                t_scores = self.transitions[:, tag]
                t_scores = t_scores.unsqueeze(0)

                # combine current scores with previous alphas
                scores = e_scores + t_scores + alphas

                # so far is exactly like the forward algorithm,
                # but now, instead of calculating the logsumexp,
                # we will find the highest score and the tag associated with it
                max_score, max_score_tag = torch.max(scores, dim=-1)

                # add the max score for the current tag
                alpha_t.append(max_score)

                # add the max_score_tag for our list of backpointers
                backpointers_t.append(max_score_tag)

            # create a torch matrix from alpha_t
            # (bs, nb_labels)
            new_alphas = torch.stack(alpha_t).t()

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas

            # append the new backpointers
            backpointers.append(backpointers_t)

        # add the scores for the final transition
        last_transition = self.transitions[:, self.EOS_TAG_ID]
        end_scores = alphas + last_transition.unsqueeze(0)

        # get the final most probable score and the final most probable tag
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)

        # find the best sequence of labels for each sample in the batch
        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        for i in range(batch_size):
            # recover the original sentence length for the i-th sample in the batch
            sample_length = emission_lengths[i].item()

            # recover the max tag for the last timestep
            sample_final_tag = max_final_tags[i].item()

            # limit the backpointers until the last but one
            # since the last corresponds to the sample_final_tag
            sample_backpointers = backpointers[: sample_length - 1]

            # follow the backpointers to build the sequence of labels
            sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)

            # add this path to the list of best sequences
            best_sequences.append(sample_path)

        return max_final_scores, best_sequences

    def _find_best_path(self, sample_id, best_tag, backpointers):
        """Auxiliary function to find the best path sequence for a specific sample.
            Args:
                sample_id (int): sample index in the range [0, batch_size)
                best_tag (int): tag which maximizes the final score
                backpointers (list of lists of tensors): list of pointers with
                shape (seq_len_i-1, nb_labels, batch_size) where seq_len_i
                represents the length of the ith sample in the batch
            Returns:
                list of ints: a list of tag indexes representing the bast path
        """

        # add the final best_tag to our best path
        best_path = [best_tag]

        # traverse the backpointers in backwards
        for backpointers_t in reversed(backpointers):
            # recover the best_tag at this timestep
            best_tag = backpointers_t[best_tag][sample_id].item()

            # append to the beginning of the list so we don't need to reverse it later
            best_path.insert(0, best_tag)

        return best_path

    def _generate_mask(self, idxs_to_collect):
        # get the mask that zeros out all the paddings
        seq_lengths = torch.sum(idxs_to_collect !=0, dim=1) - 2
        maxlen = torch.max(seq_lengths)
        mask = torch.zeros((len(seq_lengths), maxlen)).cuda()
        for i in range(len(seq_lengths)):
            mask[i, :seq_lengths[i]] = 1
        return mask

    def _getFirstSubEmbeddings(self, all_embeddings, idxs_to_collect):
        """
        Getting the embeddings of the first subwords to represent the whole words
        @return embeddings (list[tensors])
        @return pred_embeddings (list[tensors])
        """
        embeddings = []
        N = all_embeddings.shape[0]  # it's equivalent to N=len(all_embeddings)

        for i in range(N):
            collected = all_embeddings[i][idxs_to_collect[i][:-2]]  # collecting a slice of tensor
            embeddings.append(collected)
        return embeddings

    def get_fist_subword_embeddings(self, all_embeddings, idxs_to_collect):
        """
        Pick first subword embeddings with the indices list idxs_to_collect
        :param all_embeddings:
        :param idxs_to_collect:
        :param sentence_lengths:
        :return:
        """
        N = all_embeddings.shape[0]  # it's equivalent to N=len(all_embeddings)

        # Other two mode need to be taken care of the issue
        # that the last index becomes the [SEP] after argument types
        sent_embeddings = []

        for i in range(N):
            to_collect = idxs_to_collect[i]
            to_collect = to_collect[to_collect>0][:-2]
            collected = all_embeddings[i, to_collect]  # collecting a slice of tensor
            sent_embeddings.append(collected)

        max_sent_len = idxs_to_collect.shape[1]-2
        for i in range(N):
            sent_embeddings[i] = torch.cat((sent_embeddings[i], torch.zeros(max_sent_len - len(sent_embeddings[i]), self.embedding_dim).cuda()))

        sent_embeddings = torch.stack(sent_embeddings)
        return sent_embeddings

    def _get_bert_features(self, bert_sentence_batch, idxs_to_collect):
        output = self.bert(bert_sentence_batch.long())
        embeddings = output[0]
        embeds = self.get_fist_subword_embeddings(embeddings, idxs_to_collect)

        # embeds = torch.stack(w_embeds_list)
        bert_feats = self.hidden2tag(embeds)
        return bert_feats

    def neg_log_likelihood(self, tags_batch, feats, mask):
        forward_score = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(feats, tags_batch, mask)
        return torch.sum(forward_score - gold_score)

    def forward(self, sentence_batch, idxs_to_collect):
        feats = self._get_bert_features(sentence_batch, idxs_to_collect)
        mask = self._generate_mask(idxs_to_collect)
        return feats, mask

