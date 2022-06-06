# coding=utf-8
import numpy as np
from os import listdir
from os.path import isfile, join
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from torch.autograd import Variable
import random
import re
from functools import partial
from collections import defaultdict

trigger_count = [0]


def _unpack_arg_wo_vt(data, tokenizer):
    """
    read the data with pos tags
    :param data:
    :param tokenizer:
    :return:
    """
    n = len(data)
    sentence = data[0].split()
    bert_words = tokenizer.tokenize(data[0])
    triggers, arguments = [],  []
    if n > 4:
        mid = (n-2) // 2
        triggers = [x.split(' ') for x in data[1:mid]]
        arguments = [x.split(' ') for x in data[mid:-3]]
        trigger_count[0] += len(triggers)

    entities = data[-3].split(' ')
    pos = data[-2].split(' ')
    doc_id = data[-1].split(' ')

    return sentence, bert_words, triggers, arguments, entities, pos, doc_id


def _unpack_ace_with_vt(data, tokenizer):
    """
    read the data with pos tags
    :param data:
    :param tokenizer:
    :return:
    """
    n = len(data)
    sentence = data[0].split()
    bert_words = tokenizer.tokenize(data[0])
    triggers, arguments = [],  []
    if n > 6:
        mid = (n-6) // 4 + 1
        triggers = [x.split(' ') for x in data[1:mid]]
        arguments = [x.split(' ') for x in data[mid:-5]]
        trigger_count[0] += len(triggers)

    entities = data[-5].split(' ')
    values = data[-4].split(' ')
    times = data[-3].split(' ')
    pos = data[-2].split(' ')
    doc_id = data[-1].split(' ')

    return sentence, bert_words, triggers, arguments, entities, values, times, pos, doc_id


def _unpack_ere_with_filler(data, tokenizer):
    """
    read the data with pos tags with fillers
    :param data:
    :param tokenizer:
    :return:
    """
    n = len(data)
    sentence = data[0].split()
    bert_words = tokenizer.tokenize(data[0])
    triggers, arguments, filler_args = [],  [], []
    if n > 5:
        # mid = (n-5) // 3 + 1
        if (n-5) % 3 != 0:
            data = split_multiple_event_roles(data)
        if (len(data) - 5)%3!=0:
            print('error in extracting data')
            pass
        else:
            mid = (n - 5) // 3 + 1
            triggers = [x.split(' ') for x in data[1:mid]]
            arguments = [x.split(' ') for x in data[mid:-4]]
            filler_args = arguments[1::2]
            arguments = arguments[::2]
            trigger_count[0] += len(triggers)

    entities = data[-4].split(' ')
    fillers = data[-3].split(' ')
    # values = data[-4].split(' ')
    # times = data[-3].split(' ')
    pos = data[-2].split(' ')
    doc_id = data[-1].split(' ')

    return sentence, bert_words, triggers, arguments, filler_args, entities, fillers, pos, doc_id


def read_data_from(file, tokenizer, ace=True, with_vt=True):
    """
    Extract data from ACE BIO files
    :param file: path
    :return: a list [[sentence1, triggers, arguments], [sentence2, triggers, arguments], ...]
                triggers' length is equal to arguments length, and each arguments row corresponding to the trigger
    """
    data = open(file, 'r').read().split('\n\n')
    output_ = [i.split('\n') for i in data]
    output = []

    # delete the last empty string in the data file if there is one
    if output_[-1]==['']:
        output_ = output_[:-1]
    if ace and with_vt:
        output = list(map(partial(_unpack_ace_with_vt, tokenizer=tokenizer), output_))
    elif ace and not with_vt:
        output = list(map(partial(_unpack_arg_wo_vt, tokenizer=tokenizer), output_))
    elif not ace and with_vt:
        output = list(map(partial(_unpack_ere_with_filler, tokenizer=tokenizer), output_))
    elif ace and not with_vt:
        output = list(map(partial(_unpack_arg_wo_vt, tokenizer=tokenizer), output_))
    return output


def split_multiple_event_roles(data):
    ret = [data[0]]
    idxs = set()
    N, M = len(data)-4, len(data[1].split(' '))
    for i in range(1,N):
        if '#@#' in data[i]:
            idxs.add(i)

    for i in range(1,N):

        if i not in idxs:
            ret.append(data[i])
        else:
            org_line = data[i].split(' ')
            new_split = []
            event_set = set(org_line)
            event_set.remove('O')
            event_set = list(event_set)[0][2:].split('#@#')
            for this_e in event_set:
                this_event = ['O' if org_line[k]=='O' else org_line[k][0] + org_line[k][1] + this_e for k in range(M)]
                new_split.append(' '.join(this_event))
            ret.extend(new_split)
    ret.extend(data[-4:])
    return ret


def event_arg_entites_pairs(data):
    """
    Get a dictionary for event-arguments pairs
    :param data:
    :return:
    """
    event_args_dic = defaultdict(set)
    entities_arg_dic = defaultdict(set)
    all_triggers = sum([d[2] for d in data], [])
    all_args = sum([d[3] for d in data], [])
    all_entities = sum([[d[4]] * len(d[3]) for d in data if d[3]], [])

    all_triggers = list(filter(None, all_triggers))
    all_args = list(filter(None, all_args))
    all_entities = list(filter(None, all_entities))

    # event-args dic
    n = len(all_args)
    for i in range(n):
        event = list(set([i[2:] for i in all_triggers[i] if i != 'O' and len(i) > 1]))[0]
        args = set([i[2:] for i in all_args[i] if i != 'O' and len(i) > 1])
        event_args_dic[event] = event_args_dic[event] | args

    # get entity-arguments dic
    for i in range(n):
        _this_args = all_args[i][:]
        this_entities = all_entities[i][:]
        this_args = [_this_args[j] for j in range(len(_this_args)) if _this_args[j] != 'O']
        this_entities = [this_entities[j] for j in range(len(_this_args)) if _this_args[j] != 'O']
        for a, e in zip(this_args, this_entities):
            entities_arg_dic[e[2:].split(':')[0]].add(a[2:])
    return event_args_dic, entities_arg_dic


def pad_seed_sent_to_bert_sent(sent, trigger_indicator, tokenizer):
    bert_sent = []
    new_trigger_indicator = []

    sent = sent.split(' ')
    for i in range(len(sent)):
        tmp = tokenizer.tokenize(sent[i].replace('/\'', '\''))
        bert_sent.extend(tmp)

        new_trigger_indicator.extend([trigger_indicator[i]] * len(tmp))

    new_trigger_indicator = list(np.nonzero(new_trigger_indicator)[0])
    return bert_sent, new_trigger_indicator


def data_extract_trigger(data, shuffle=False):
    index_array = list(range(len(data)))
    if shuffle:
        np.random.shuffle(index_array)

    indices = index_array[:]
    data = [data[idx] for idx in indices]

    sent_indx = [e[0] for e in data]
    words = [e[1] for e in data]
    bert_words = [e[2] for e in data]
    event_queries = [e[3] for e in data]
    event_tags = [e[4] for e in data]
    pos_tags = [e[5] for e in data]
    entity_bio = [e[6] for e in data]

    return sent_indx, words, bert_words, event_queries, event_tags, pos_tags, entity_bio


def prepare_bert_sequence(seq_batch, to_ix, pad, emb_len):
    padded_seqs = []
    for seq in seq_batch:
        pad_seq = torch.full((emb_len,), to_ix(pad), dtype=torch.int)
        # ids = [to_ix(w) for w in seq]
        ids = to_ix(seq)
        pad_seq[:len(ids)] = torch.tensor(ids, dtype=torch.long)
        padded_seqs.append(pad_seq)
    return torch.stack(padded_seqs)


def firstSubwordsIdx_for_one_seq(words, tokenizer, prompt=False):
    """
    extract first subword indices for one sentence
    :param words:
    :param tokenizer:
    :return:
    """
    if not prompt:
        collected_1st_subword_idxs = []
        idx = 1  # need to skip the embedding of '[CLS]'
        has_seen_sep_at = 0
        for i in range(len(words)):

            w = words[i]
            if words[i] == '[SEP]':
                has_seen_sep_at = i
                collected_1st_subword_idxs.append(has_seen_sep_at)
                break

        idx = has_seen_sep_at + 1
        for i in range(has_seen_sep_at + 1, len(words)):
            w_tokenized = tokenizer.tokenize(words[i])
            collected_1st_subword_idxs.append(idx)
            idx += len(w_tokenized)
    else:
        collected_1st_subword_idxs = []
        idx = 1  # need to skip the embedding of '[CLS]'

        for i in range(1, len(words)):
            if words[i] == '[SEP]':
                break
            w_tokenized = tokenizer.tokenize(words[i])
            collected_1st_subword_idxs.append(idx)
            idx += len(w_tokenized)

    return collected_1st_subword_idxs


def firstSubwordsIdx_batch(seq_batch, tokenizer, prompt=False):
    """
    extract first subword indices for one batch
    :param seq_batch:
    :param tokenizer:
    :return:
    """
    idx_batch = []
    for seq in seq_batch:
        idx_seq = firstSubwordsIdx_for_one_seq( seq, tokenizer, prompt )
        idx_batch.append(idx_seq)
    return idx_batch


def prepare_sequence(seq_batch, to_ix, pad, seqlen, remove_bio_prefix=False):
    padded_seqs = []
    for seq in seq_batch:
        if pad == -1:
            pad_seq = torch.full((seqlen,), pad, dtype=torch.int)
        else:
            pad_seq = torch.full((seqlen,), to_ix[pad], dtype=torch.int)
        if remove_bio_prefix:
            ids = [to_ix[w[2:]] if len(w) > 1 and w[2:] in to_ix else to_ix['O'] for w in seq]
        else:
            ids = [to_ix[w] if w in to_ix else -1 for w in seq ]

        pad_seq[:len(ids)] = torch.Tensor(ids).long()
        padded_seqs.append(pad_seq)
    return torch.stack(padded_seqs)


def pad_trigger_pack(seq_batch, to_ix, pad, seqlen, join_with_ere=False):
    padded_seqs = []
    n_trigger = [33 if not join_with_ere else 38][0]
    for seq in seq_batch:
        if pad == -1:
            pad_seq = torch.full((n_trigger, seqlen), pad, dtype=torch.int)
        else:
            pad_seq = torch.full((n_trigger, seqlen), to_ix[pad], dtype=torch.int)
        ids = [[to_ix[w] if w in to_ix else -1 for w in x] for x in seq]
        pad_seq[:len(ids), :len(ids[0])] = torch.Tensor(ids).long()

        padded_seqs.append(pad_seq)
    return torch.stack(padded_seqs)


def generate_entity_identifier(entity_bio, seq_length, PAD_TAG=-1,):
    padded_seqs = []
    for i in range(len(entity_bio)):
        this_tags = entity_bio[i]
        tmp = -1
        pad_seq = torch.full((seq_length,), PAD_TAG, dtype=torch.int)
        this_entity = ''
        for j in range(len(this_tags)):
            if this_tags[j][0] == 'B':
                tmp += 1
                pad_seq[j] = tmp
                this_entity = this_tags[j][2:]
            elif this_tags[j][0] == 'I':
                if this_entity == this_tags[j][2:]:
                    pad_seq[j] = tmp
        padded_seqs.append(pad_seq)
    return torch.stack(padded_seqs).long()


def pad_tensor(X, new_dim=38):
    shape = torch.tensor(X.shape).tolist()
    shape[1] = new_dim
    tmp = torch.zeros(shape)
    tmp[:, :X.shape[1]] = X
    return tmp


def data_prepare(data_bert, tokenizer, config, word_to_ix, trigger_to_ix, entity_to_ids,
                 join_with_ere=False, dataset_id=0):
    """
    Generate data loader for the model
    :param data_bert:
    :param tokenizer:
    :param config:
    :param word_to_ix:
    :param trigger_to_ix:
    :return:
    """
    # data preparation
    doc_id, words_batch, bert_words_batch, event_queries, trigger_tags_batch, pos_tags, entity_bio_ = data_extract_trigger(data_bert)
    sentence_lengths = list(map(len, words_batch))  # [len(s) for s in words_batch]
    seq_length = max(sentence_lengths)
    bert_sentence_lengths = list(map(len, bert_words_batch))  # [len(s) for s in bert_words_batch]
    bert_seq_length = max(bert_sentence_lengths)
    # entity_tag = prepare_sequence(entity_bio_[:], entity_to_ids, config.PAD_TAG, seq_length, remove_bio_prefix=True).long()
    entity_identifier = generate_entity_identifier(entity_bio_, seq_length)

    idxs_to_collect_batch = firstSubwordsIdx_batch(words_batch, tokenizer)

    bert_tokens = prepare_bert_sequence(bert_words_batch, word_to_ix, config.PAD_TAG, bert_seq_length)
    trigger_tags = pad_trigger_pack(trigger_tags_batch, trigger_to_ix, config.PAD_TAG, seq_length, join_with_ere)

    # seed_trigger_idx = pad_sequences(seed_trigger_idx, maxlen=9, dtype="long", truncating="post",padding="post")

    first_subword_idxs = pad_sequences(idxs_to_collect_batch, maxlen=seq_length+2, dtype="long", truncating="post",padding="post")
    first_subword_idxs = torch.Tensor(first_subword_idxs).long()
    sent_lengths = torch.Tensor(sentence_lengths).unsqueeze(1).long()
    doc_id = torch.Tensor(doc_id).unsqueeze(1).long()
    pos_tags = prepare_sequence(pos_tags, config.pos2id, -1, seq_length)
    pos_tags = pos_tags.cuda()
    doc_id, bert_tokens, trigger_tags, first_subword_idxs,\
    sent_lengths, bert_sentence_lengths = \
    doc_id.cuda(), bert_tokens.cuda(), trigger_tags.cuda(), first_subword_idxs.cuda(),\
    sent_lengths.cuda(), torch.Tensor(bert_sentence_lengths).cuda()
    entity_identifier = entity_identifier.cuda()

    dataset_id = dataset_id * torch.ones(doc_id.shape[0], 1).cuda()
    return dataset_id, doc_id, bert_tokens, trigger_tags, first_subword_idxs,\
           sent_lengths, bert_sentence_lengths, pos_tags, entity_identifier
           # map_numerator.cuda(), event_queries.cuda(), event_query_map.cuda()


def generate_entity_mask(entity_tags, sent_lengths, entity_arg_ids_dic, arg_types=22):
    h, w = entity_tags.shape
    mask = torch.zeros(h, w, arg_types+1)  # plus 1 dim for 'O'
    # for x in range(h):
    #     for y in range(sent_lengths[x]):
    #         mask[x, y, list(entity_arg_ids_dic[entity_tags[x, y].item()])] = 1
    mask = [generate_entity_mask_single(w, entity_tags[x], sent_lengths[x], entity_arg_ids_dic, arg_types) for x in range(h)]
    mask = torch.stack(mask)
    mask[:, :, -1] = 1
    return mask


def generate_entity_mask_single(w, entity_tags, sent_length, entity_arg_ids_dic, arg_types=22):
    mask = torch.zeros(w, arg_types + 1)
    for y in range(sent_length):
        mask[y, list(entity_arg_ids_dic[entity_tags[y].item()])] = 1
    return mask


def firstSubwordsIdx_for_one_seq_arg(words, tokenizer):
    """
    extract first subword indices for one sentence
    :param words:
    :param tokenizer:
    :return:
    """
    collected_1st_subword_idxs = []
    idx = 1  # need to skip the embedding of '[CLS]'
    for i in range(1, len(words)):
        w = words[i]
        w_tokenized = tokenizer.tokenize(w)
        collected_1st_subword_idxs.append(idx)
        idx += len(w_tokenized)
        if w == '[SEP]':
            break

    bert_len = len(tokenizer.tokenize(' '.join(words)))
    collected_1st_subword_idxs.append(bert_len-1)
    return collected_1st_subword_idxs


def firstSubwordsIdx_batch_arg(seq_batch, tokenizer):
    """
    extract first subword indices for one batch
    :param seq_batch:
    :param tokenizer:
    :return:
    """
    idx_batch = []
    for seq in seq_batch:
        idx_seq = firstSubwordsIdx_for_one_seq_arg(seq, tokenizer)
        idx_batch.append(idx_seq)
    return idx_batch


def bio_to_ids(bio_tags, tags_to_ids, remove_overlap=False, is_trigger=False, is_entity=False):
    if remove_overlap:
        bio_tags = [[x.split('#@#')[0] for x in args] for args in bio_tags]
    if is_trigger:
        arg_remove_bio = [[tags_to_ids[x[2:]] if len(x) > 2 else tags_to_ids[x] for x in args] for args in bio_tags]
    elif is_entity:
        arg_remove_bio = [[tags_to_ids[re.split('[: -]', x[2:])[0]] if len(x) > 2 else tags_to_ids[x] for x in args] for args in bio_tags]
    else:
        arg_remove_bio = [[tags_to_ids[re.split('[-#]', x[2:])[0]] if len(x) > 2 and x[2:] in tags_to_ids else tags_to_ids['O'] for x in args] for args in bio_tags]
    args = torch.Tensor(pad_sequences(arg_remove_bio, dtype="long", truncating="post", padding="post", value=tags_to_ids['[PAD]']))
    return args.long()


def pack_data_to_trigger_model(batch, config):
    """
    Prepare data for trigger detection model
    :param batch:
    :param config:
    :return:
    """
    # unpack and truncate data
    sent_indx, bert_sentence_in, trigger_tags, list_idxs_to_collect,\
    sent_lengths, bert_sentence_lengths, pos_tags, entity_identifier = batch
    embedding_length = int(torch.max(bert_sentence_lengths).item())
    bert_sentence_in = bert_sentence_in[:, :embedding_length]
    max_sent_len = int(torch.max(sent_lengths).item())
    list_idxs_to_collect = list_idxs_to_collect[:, :max_sent_len + 2].long()
    trigger_tags = trigger_tags[:, :, :max_sent_len]
    pos_tags = pos_tags[:, :max_sent_len].long()
    entity_identifier = entity_identifier[:, :max_sent_len]

    # send data to gpu
    sent_lengths = sent_lengths.long()
    # seed_trigger_idx = Variable(seed_trigger_idx)
    if config.use_gpu:
        bert_sentence_in, trigger_tags, list_idxs_to_collect, sent_lengths, \
        sent_indx, bert_sentence_lengths, pos_tags \
            = bert_sentence_in.to(config.device), trigger_tags.to(config.device), \
              list_idxs_to_collect.to(config.device), sent_lengths.to(config.device), \
              sent_indx.to(config.device), \
              bert_sentence_lengths.to(config.device), pos_tags.to(config.device)
        # seed_trigger_idx = seed_trigger_idx.to(config.device)
        entity_identifier = entity_identifier.to(config.device)

    return bert_sentence_in, trigger_tags, list_idxs_to_collect, sent_lengths, embedding_length, \
           sent_indx, bert_sentence_lengths, pos_tags, entity_identifier


def pack_data_to_trigger_model_joint(batch, config):
    """
    Prepare data for trigger detection model
    :param batch:
    :param config:
    :return:
    """
    # unpack and truncate data
    dataset_id, sent_indx, bert_sentence_in, trigger_tags, list_idxs_to_collect,\
                    sent_lengths, bert_sentence_lengths, pos_tags, entity_identifier = batch
    embedding_length = int(torch.max(bert_sentence_lengths).item())
    bert_sentence_in = bert_sentence_in[:, :embedding_length]
    max_sent_len = int(torch.max(sent_lengths).item())
    list_idxs_to_collect = list_idxs_to_collect[:, :max_sent_len + 2].long()
    trigger_tags = trigger_tags[:, :, :max_sent_len]
    pos_tags = pos_tags[:, :max_sent_len].long()
    entity_identifier = entity_identifier[:, :max_sent_len]
    entity_identifier = Variable(entity_identifier)

    # send data to gpu
    # sent_lengths = sent_lengths.long()
    sent_lengths = torch.sum(list_idxs_to_collect!=0, dim=-1) - 2
    bert_sentence_in, list_idxs_to_collect, sent_lengths = \
        Variable(bert_sentence_in), \
        Variable(list_idxs_to_collect), Variable(sent_lengths.long())
    sent_indx, bert_sentence_lengths = \
        Variable(sent_indx), Variable(bert_sentence_lengths)
    trigger_tags = Variable(trigger_tags)
    pos_tags = Variable(pos_tags)

    return dataset_id, bert_sentence_in.long(), trigger_tags, list_idxs_to_collect, sent_lengths.long(), embedding_length, \
           sent_indx, bert_sentence_lengths, pos_tags, entity_identifier


def firstSubwordsIdx_for_one_seq_template(words, tokenizer):
    """
    extract first subword indices for one sentence
    :param words:
    :param tokenizer:
    :return:
    """
    each_token_len = [len(tokenizer.tokenize(w)) for w in words ]
    collected_1st_subword_idxs = [sum(each_token_len[:x+1]) for x in range(len(words)-1) if words[x+1] != '[SEP]']
    return collected_1st_subword_idxs


def firstSubwordsIdx_batch_template(seq_batch, tokenizer, is_template=True):
    """
    extract first subword indices temfor one batch
    :param seq_batch:
    :param tokenizer:
    :return:
    """
    all_idx_to_collect = [firstSubwordsIdx_for_one_seq_template(seq, tokenizer) for seq in seq_batch]
    all_sent = [' '.join(x[1:]) for x in seq_batch]
    if is_template:
        event_prefix_wo_def_len = [' '.join(x.split(' [SEP] ')[1:3]) for x in all_sent]
    else:
        event_prefix_wo_def_len = [' '.join(x.split(' [SEP] ')[1:]) for x in all_sent]
    event_prefix_wo_def_len = [len(x.split()) for x in event_prefix_wo_def_len]

    sent_len = [' '.join(x.split(' [SEP] ')[:1]) for x in all_sent]
    sent_len = [len(x.split()) for x in sent_len]

    sent_idx_to_collect = [all_idx_to_collect[x][:sent_len[x]] for x in range(len(all_idx_to_collect))]
    event_idx_to_collect = [all_idx_to_collect[x][sent_len[x]:sent_len[x]+event_prefix_wo_def_len[x]] for x in range(len(all_idx_to_collect))]
    return sent_idx_to_collect, event_idx_to_collect


def extract_trigger_for_prompt(trigger_bio, sent_len, event_to_ids):
    trigger_bio = [sorted(i)[0] for i in zip(*trigger_bio)]
    if not trigger_bio:
        return [event_to_ids['O']] * sent_len
    trigger_bio = [x[2:] if len(x) > 2 else x for x in trigger_bio ]

    return [event_to_ids[x] for x in trigger_bio]


def data_extract_prompt(data, shuffle=False):
    index_array = list(range(len(data)))
    if shuffle:
        np.random.shuffle(index_array)

    indices = index_array[:]
    data = [data[idx] for idx in indices]

    doc_id = [e[0] for e in data]
    sent_id = [e[1] for e in data]
    words_batch = [e[2] for e in data]
    bert_words_batch = [e[3] for e in data]
    event_bio = [e[4] for e in data]
    pos_tag = [e[5] for e in data]
    token_type_ids = [e[6] for e in data]
    # candidate_ids = [e[7] for e in data]

    return doc_id, sent_id, words_batch, bert_words_batch, event_bio, \
           pos_tag, token_type_ids


def data_extract_long_trigger(data, shuffle=False):
    index_array = list(range(len(data)))
    if shuffle:
        np.random.shuffle(index_array)

    indices = index_array[:]
    data = [data[idx] for idx in indices]

    doc_id = [e[0] for e in data]
    words_batch = [e[1] for e in data]
    bert_words_batch = [e[2] for e in data]
    event_type = [e[3] for e in data]
    event_bio = [e[4] for e in data]
    new_arg_bio = [e[5] for e in data]
    # value_arg_bio = [e[6] for e in data]
    # time_arg_bio = [e[7] for e in data]
    pos_tag = [e[6] for e in data]
    entity_bio = [e[7] for e in data]

    return doc_id, words_batch, bert_words_batch, event_type, event_bio, new_arg_bio, \
           pos_tag, entity_bio


def event_arg_to_query_and_mask(trigger_list, event_arg_dic, arg_to_token_ids, args_to_type_ids):
    """
    For each event, compose an argument query and an argument masking
    :param event_arg_dic:
    :param arg_to_token_ids:
    :return:
    """
    ret = dict()
    max_w = max([len(v) for v in event_arg_dic.values()])
    queries = [sum([arg_to_token_ids[arg] for arg in v], []) for v in event_arg_dic.values()]
    queries = pad_sequences(queries, dtype="long", truncating="post", padding="post")
    max_h = queries.shape[1]

    for event in trigger_list:
        begin, end = 0, 0
        j = 0
        this_query = []
        # extra slot for [SEP] to indicate 'O'
        this_mask = np.zeros((max_h+1, max_w+1))
        arg_ids = []
        for arg in event_arg_dic[event]:
            this_query.append(arg_to_token_ids[arg])
            end += len(arg_to_token_ids[arg])
            this_mask[begin:end, j] = 1 / (end - begin)
            j += 1
            begin = end
            arg_ids.append(args_to_type_ids[arg])
        arg_ids.append(args_to_type_ids['O'])
        this_query = np.array(sum(this_query, []), dtype=int)
        this_mask[end, j] = 1
        ret[event] = (this_query, this_mask, arg_ids)
    return ret


def arg_to_token_ids(arg_set, tokenizer):
    """
    Generate a dictionary for argument with token ids
    :param arg_set: sorted arg set
    :param tokenizer:
    :return:
    """
    ret = dict()
    for x in arg_set:
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))
        ret[x] = ids
    return ret


def trigger_arg_bio_to_ids(trigger_bio, arg_bio, event_type, sent_len):
    ret = defaultdict(list)

    # ner_arg = arg_bio[::3]
    # value_arg = arg_bio[1::3]
    # time_arg = arg_bio[2::3]
    if trigger_bio:
        N = len(trigger_bio)
        for i in range(N):
            this_trigger = set(trigger_bio[i])
            this_trigger.remove('O')
            if this_trigger:
                this_trigger = list(this_trigger)[0][2:]
            # ret[this_trigger].append([trigger_bio[i], ner_arg[i], value_arg[i], time_arg[i]])
            ret[this_trigger].append([trigger_bio[i], arg_bio[i]])

    no_this_trigger = ['O'] * sent_len
    for i in event_type:
        if not ret[i]:
            # ret[i] = [(no_this_trigger, no_this_trigger, no_this_trigger, no_this_trigger)]
            ret[i] = [(no_this_trigger, no_this_trigger)]

    return ret


def firstSubwordsIdx_for_one_seq_arg_post(words, tokenizer):
    """
    extract first subword indices for one sentence
    :param words:
    :param tokenizer:
    :return:
    """
    collected_1st_subword_idxs = []
    idx = 1  # need to skip the embedding of '[CLS]'
    not_seen = True
    for i in range(1, len(words)):

        w = words[i]
        if w != '[SEP]' and not_seen:
            continue
        if w == '[SEP]' and not not_seen:
            break
        elif w == '[SEP]':
            not_seen = False
            idx = i+1
            continue

        w_tokenized = tokenizer.tokenize(w)
        collected_1st_subword_idxs.extend([1/len(w_tokenized)] * len(w_tokenized))
        # idx += len(w_tokenized)

    # collected_1st_subword_idxs.append(idx)
    return collected_1st_subword_idxs


def firstSubwordsIdx_batch_arg_post(seq_batch, tokenizer):
    """
    extract first subword indices for one batch
    :param seq_batch:
    :param tokenizer:
    :return:
    """
    idx_batch = []
    # for seq in seq_batch:
    idx_batch = list(map(partial(firstSubwordsIdx_for_one_seq_arg_post, tokenizer=tokenizer), seq_batch))
    # idx_batch.append(idx_seq)
    return idx_batch

