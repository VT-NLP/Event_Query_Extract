import torch
import torch.nn as nn
from transformers import DistilBertModel
from transformers import BertModel
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np
from torch.autograd import Variable

class TriggerDetection(nn.Module):
    def __init__(self, config):
        super(TriggerDetection, self).__init__()
        self.config = config
        self.embedding_dim = config.EMBEDDING_DIM
        self.sqrt_d = np.sqrt(self.embedding_dim)
        self.entity_emb_dim = config.ENTITY_EMB_DIM

        self.device = config.device
        self.dropout_rate = 0.2
        self.pretrained_weights = str(config.pretrained_weights)
        self.same_bert = config.same_bert

        # load pre-trained embedding layer
        if self.pretrained_weights == 'distilbert-base-uncased':
            self.bert = DistilBertModel.from_pretrained(self.pretrained_weights)
        elif self.pretrained_weights == 'bert':
            self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
            if not self.same_bert:
                self.bert_for_event_type = BertModel.from_pretrained('bert-base-uncased')
        elif self.pretrained_weights == 'bertlarge':
            self.bert = BertModel.from_pretrained('bert-large-uncased', output_attentions=True)
            if not self.same_bert:
                self.bert_for_event_type = BertModel.from_pretrained('bert-large-uncased')
        elif self.pretrained_weights == 'spanbert':
            self.bert = AutoModel.from_pretrained("SpanBERT/spanbert-large-cased")
            if not self.same_bert:
                self.bert_for_event_type = BertModel.from_pretrained("SpanBERT/spanbert-large-cased")
        else:
            self.bert = None
        if config.freeze_bert:
            for name, param in self.bert.named_parameters():
                param.requires_grad = False

        n_hid = 500
        n_hid2 = 100
        self.linear_bert1 = nn.Linear(self.embedding_dim, n_hid)
        self.linear_bert2 = nn.Linear(self.embedding_dim, n_hid)
        self.linear = nn.Sequential(
            nn.Dropout(p=self.config.dropout),
            nn.Linear(self.embedding_dim*3,n_hid2),
            nn.GELU(),
        )
        self.linear2 = nn.Linear(n_hid2+17, 2)
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

        # event queries for ACE and ERE
        self.event_queries = self.config.fact_container.event_queries.cuda()
        self.event_query_map = self.config.fact_container.event_query_map.cuda()
        self.event_queries_ere = self.config.fact_container.event_queries_ere.cuda()
        self.event_query_map_ere = self.config.fact_container.event_query_map_ere.cuda()

        # numerator for different number of event type queries
        self.map_numerator = 1/torch.sum(self.event_query_map, dim=-1).float().unsqueeze(-1).unsqueeze(-1)
        self.map_numerator_ere = 1/torch.sum(self.event_query_map_ere, dim=-1).float().unsqueeze(-1).unsqueeze(-1)

    def get_fist_subword_embeddings(self, all_embeddings, idxs_to_collect, embed_lengths):
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

    def forward(self, sentence_batch, idxs_to_collect, embed_lengths, pos_tag, ere=False):
        # bert embeddings
        if ere:
            event_queries = self.event_queries_ere
            event_query_map = self.event_query_map_ere
            map_numerator = self.map_numerator_ere
        else:
            map_numerator = self.map_numerator
            event_queries = self.event_queries
            event_query_map = self.event_query_map
        idxs_to_collect[idxs_to_collect!=0] += 2
        attention_mask = (sentence_batch!=0)*1
        all_embeddings, _, hidden_layer_att = self.bert(sentence_batch, attention_mask=attention_mask)
        all_embeddings = all_embeddings * attention_mask.unsqueeze(-1)

        # pick embeddings of sentence and event type 1
        # use two separate bert for event type
        sent_embeddings = self.get_fist_subword_embeddings(all_embeddings, idxs_to_collect, embed_lengths)

        # use BERT hidden logits to calculate contextual embedding
        last0_layer_atten = self.select_hidden_att(hidden_layer_att[-1], idxs_to_collect)
        last1_layer_atten = self.select_hidden_att(hidden_layer_att[-2], idxs_to_collect)
        last2_layer_atten = self.select_hidden_att(hidden_layer_att[-3], idxs_to_collect)

        ave_layer_att = (last0_layer_atten + last1_layer_atten + last2_layer_atten)/3
        context_emb = ave_layer_att.matmul(sent_embeddings)

        query_embed_map = (event_queries!=0)*1
        event_embedding = self.bert_for_event_type(event_queries, attention_mask=query_embed_map)[0]
        event_embedding = event_embedding * event_query_map.unsqueeze(-1)

        logits = sent_embeddings.unsqueeze(1).matmul(event_embedding.permute([0, 2, 1])) / self.sqrt_d * map_numerator
        sent2event_att = logits.matmul(event_embedding)
        sent_embeddings = sent_embeddings.unsqueeze(1).repeat(1,len(event_queries),1,1)
        context_emb = context_emb.unsqueeze(1).repeat(1,len(event_queries),1,1)

        _logits = self.linear(torch.cat((sent_embeddings, sent2event_att, context_emb), dim=-1))

        pos_tag[pos_tag==-1] = 16
        pos_tag = torch.nn.functional.one_hot(pos_tag, num_classes=17)
        pos_tag = pos_tag.unsqueeze(1).repeat(1, _logits.shape[1], 1, 1)

        _logits = self.linear2(torch.cat((_logits, pos_tag), dim=-1))

        return _logits

    def select_hidden_att(self, hidden_att, idxs_to_collect):
        """
        Pick attentions from hidden layers
        :param hidden_att: of dimension (batch_size, embed_length, embed_length)
        :return:
        """
        N = hidden_att.shape[0]
        sent_len = idxs_to_collect.shape[1]-2
        hidden_att = torch.mean(hidden_att, 1)
        hidden_att_selected = torch.zeros(N, sent_len, sent_len).cuda()

        for i in range(N):
            to_collect  = idxs_to_collect[i]
            to_collect = to_collect[to_collect>0][:-2]
            collected = hidden_att[i, to_collect][:,to_collect]  # collecting a slice of tensor
            hidden_att_selected[i, :len(to_collect), :len(to_collect)] = collected

        return hidden_att_selected/(torch.sum(hidden_att_selected, dim=-1, keepdim=True)+1e-9)

    def rand_init_hidden(self, batch_size):
        """
        random initialize hidden variable
        """
        return Variable(
            torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)), Variable(
            torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim))