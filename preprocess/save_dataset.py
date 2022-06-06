import sys
sys.path.append("../")
sys.path.append("../../")
import torch
import argparse
from utils.data_to_dataloader import read_data_from
from utils.config import Config
from utils.metadata import Metadata
from collections import defaultdict
import spacy
from spacy.tokenizer import Tokenizer as spacy_tokenizer
from random import shuffle
from utils.data_to_dataloader import prepare_bert_sequence, pad_sequences, \
    bio_to_ids, firstSubwordsIdx_for_one_seq_template, prepare_sequence
import json
import numpy as np
from torch.utils.data import TensorDataset


spacy_tagger = spacy.load("en_core_web_lg")
spacy_tagger.tokenizer = spacy_tokenizer(spacy_tagger.vocab)


def token_to_berttokens(sent, tokenizer, omit=True, template=False):
    '''
    Generate Bert subtokens, and return first subword index list
    :param sent:
    :param tokenizer:
    :param omit:
    :param template:
    :return:
    '''
    bert_tokens = [tokenizer.tokenize(x) for x in sent]
    to_collect = [[1] + [0 for i in range(len(x[1:]))] for x in bert_tokens]
    if template:
        second_sep_idx = [i for i in range(len(sent)) if sent[i] == '[SEP]'][-1]
        bert_tokens_prefix = [tokenizer.tokenize(x) for x in sent[:second_sep_idx+1]]
        to_collect = [[1] + [0 for i in range(len(x[1:]))] for x in bert_tokens_prefix]
    bert_tokens = sum(bert_tokens, [])
    to_collect = sum(to_collect, [])
    if omit:
        omit_dic = {'[SEP]', '[CLS]'}
        to_collect = [x if y not in omit_dic else 0 for x, y in zip(to_collect, bert_tokens)]
    return bert_tokens, to_collect


def pair_trigger_template(data_bert, event_template, config):
    '''
    Pair trigger with event type query
    :param data_bert:
    :param event_template:
    :param config:
    :return:
    '''
    data_bert_new = []
    event_types = sorted(list(event_template.keys()))
    event_template_bert = {}
    for e in event_types:
        temp = event_template[e].split('-')
        event_template_bert[e] = token_to_berttokens(temp, config.tokenizer, template=True)

    for j in range(len(data_bert)):

        tokens, event_bio, arg_bio = data_bert[j]
        if len(tokens) > 256:
            continue
        bert_tokens, to_collect = token_to_berttokens(tokens, config.tokenizer, False)
        pos_tag = get_pos(tokens)
        if set(event_bio[0]) == {'O'}:
            event_bio, arg_bio = [], []
        trigger_arg_dic = trigger_arg_bio_to_ids(event_bio, arg_bio, event_types, len(tokens))

        for event_type in event_types:
            this_template = event_template[event_type].split('-')
            this_template_bert, this_template_to_collect = event_template_bert[event_type]
            this_tokens = this_template + tokens + ['[SEP]']

            this_trigger_bio = [x[0] for x in trigger_arg_dic[event_type]]
            this_ner_arg_bio = [x[1] for x in trigger_arg_dic[event_type]]

            bert_sent = this_template_bert + bert_tokens[:] + ['[SEP]']

            sent_idx_to_collect = [0 for _ in range(len(this_template_bert))] + to_collect[:] + [0]
            data_tuple = (this_tokens, event_type, this_trigger_bio,
                          this_ner_arg_bio, pos_tag, bert_sent, this_template_to_collect, sent_idx_to_collect)

            data_bert_new.append(data_tuple)
    return data_bert_new


def trigger_arg_bio_to_ids(trigger_bio, arg_bio, event_type, sent_len):
    '''
    Convert list annotation to dictionary
    :param trigger_bio: Trigger list [[trigger_for_event_mention1], [trigger_for_event_mention2]]
    :param arg_bio: [[args_for_event_mention1], [args_for_event_mention2]]
    :param event_type: event type list
    :param sent_len:
    :return:
    '''
    ret = defaultdict(list)

    if trigger_bio:
        N = len(trigger_bio)
        for i in range(N):
            this_trigger = set(trigger_bio[i])
            this_trigger.remove('O')
            if this_trigger:
                this_trigger = list(this_trigger)[0][2:]
            ret[this_trigger].append([trigger_bio[i], arg_bio[i]])

    no_this_trigger = ['O'] * sent_len
    for i in event_type:
        if not ret[i]:
            ret[i] = [(no_this_trigger, no_this_trigger)]

    return ret


def get_pos(sentence):
    '''
    Get POS tag for input sentence
    :param sentence:
    :return:
    '''
    doc = spacy_tagger(' '.join(list(sentence)))
    ret = []
    for token in doc:
        ret.append(token.pos_)
    return ret


def remove_irrelevent_data(data, ere=False):
    '''
    Keep input sentence, trigger annotations and argument annotations
    :param data:
    :param ere:
    :return:
    '''
    to_collect_idx = [0, 2, 3]
    data = [[x[y] for y in to_collect_idx] for x in data]
    for x in data:
        if not ere:
            x[2] = x[2][::3]  # discard value and time arguments
        if x[1] == []:
            x[1] = [['O' for _ in range(len(x[0]))]]
            x[2] = [['O' for _ in range(len(x[0]))]]
    return data


def data_extract(data, shuffle=False):
    index_array = list(range(len(data)))
    if shuffle:
        np.random.shuffle(index_array)

    indices = index_array[:]
    data = [data[idx] for idx in indices]

    words_batch = [e[0] for e in data]
    event_type = [e[1] for e in data]
    event_bio = [e[2] for e in data]
    arg_bio = [e[3] for e in data]
    pos_tag = [e[4] for e in data]
    bert_tokens = [e[5] for e in data]
    idxs_to_collect_sent = [e[6] for e in data]
    idxs_to_collect_event = [e[7] for e in data]

    return words_batch, event_type, event_bio, arg_bio, pos_tag, bert_tokens, idxs_to_collect_sent, idxs_to_collect_event


def dataset_prepare_trigger_zero_template(data_bert, config, trigger_to_ids, metadata):
    """
    Generate data loader for the argument model
    :param data_bert:
    :param config:
    :param trigger_to_ids:
    :param metadata
    :return:
    """
    # unpack data
    word_to_ix = config.tokenizer.convert_tokens_to_ids

    tokens, event_type, event_bio, _, pos_tag, bert_sent, idxs_to_collect_event, idxs_to_collect_sent = data_extract(
        data_bert)

    # general information: sent_len, bert_sent_len, first_index
    bert_sentence_lengths = [len(s) for s in bert_sent]
    max_bert_seq_length = int(max(bert_sentence_lengths))
    sentence_lengths = [len(x) for x in tokens]
    max_seq_length = int(max(sentence_lengths))
    bert_tokens = prepare_bert_sequence(bert_sent, word_to_ix, config.PAD_TAG, max_bert_seq_length)
    # general information: pad_sequence
    idxs_to_collect_sent = pad_sequences(idxs_to_collect_sent, dtype="long", truncating="post", padding="post")
    idxs_to_collect_sent = torch.Tensor(idxs_to_collect_sent)
    idxs_to_collect_event = pad_sequences(idxs_to_collect_event, dtype="long", truncating="post", padding="post")
    idxs_to_collect_event = torch.Tensor(idxs_to_collect_event)

    sent_lengths = torch.Tensor(sentence_lengths).unsqueeze(1)
    pos_tags_all = prepare_sequence(pos_tag, metadata.pos2id, -1, max_seq_length)
    bert_sentence_lengths = torch.Tensor(bert_sentence_lengths)

    # trigger
    for i in range(len(event_bio)):
        if len(event_bio[i]) == 1:
            event_bio[i] = event_bio[i][0]
        else:
            event_bio[i] = [min(np.array(event_bio[i])[:, j]) for j in range(len(event_bio[i][0]))]
    event_tags = bio_to_ids(event_bio, trigger_to_ids, is_trigger=True)

    long_data = (sent_lengths, bert_sentence_lengths,
                 bert_tokens, idxs_to_collect_event, idxs_to_collect_sent, pos_tags_all, event_tags)
    return long_data


def get_event_rep(f='../utils/trigger_representation_ace.json', rep='type_name_seed_template'):
    f = open(f, 'r')
    trigger_representation_json = json.load(f)
    f.close()
    return trigger_representation_json[rep]['suppliment_trigger']


def save_trigger_dataset(dataset, path=None):
    dataset = [x.cuda() for x in dataset]
    tensor_set = TensorDataset(*dataset)
    if path:
        torch.save(tensor_set, path)
    print('save file to ', path)
    return 0


def save_to_json(data, file):
    res = []
    for x in data:
        event_list = []
        arg_list = []
        sentence, triggers, args = x
        sent_len = len(sentence)
        if set(triggers[0]) == {'O'}:
            res.append({'event_trigger': [], 'arg_list': []})
            continue
        for k in range(len(triggers)):
            trigger_ids = [i for i in range(sent_len) if triggers[k][i] != 'O']
            event_begin, event_end = trigger_ids[0], trigger_ids[-1] + 1
            event_type = triggers[k][event_begin][2:]
            arg_begins = [i for i in range(sent_len) if args[k][i][0] == 'B']
            arg_types = [args[k][i][2:] for i in range(sent_len) if args[k][i][0] == 'B']
            arg_ends = []
            for a in arg_begins:
                b = a+1
                while b<sent_len:
                    if args[k][b][0] == 'I':
                        b+=1
                        continue
                    else:
                        break
                arg_ends.append(b)
            arg_list = [(event_type, x, y, z) for x, y, z in zip(arg_types, arg_begins, arg_ends)]
            event_list.append([event_type, event_begin, event_end])
        res.append({'event_trigger': event_list, 'arg_list': arg_list})
    jsonString = json.dumps(res)
    jsonFile = open(file, "w")
    jsonFile.write(jsonString)
    jsonFile.close()
    print('save to ', file)
    return res


def read_from_source(args):
    config = Config()
    torch.backends.cudnn.enabled = False
    torch.manual_seed(39)

    metadata = Metadata()
    e_rep = 'event_name_seed'
    if args.ace:
        data_folder = args.data_folder
        trigger_to_ids = metadata.ace.triggers_to_ids
        event_rep = get_event_rep(config.project_root + '/preprocess/ace/trigger_representation.json', e_rep)
        save_path = config.project_root + '/data/ace_en/pt/'
    else:
        data_folder = args.data_folders
        trigger_to_ids = metadata.ere.triggers_to_ids
        event_rep = get_event_rep(config.project_root + '/preprocess/ere/trigger_representation.json', e_rep)
        save_path = config.project_root + '/data/ere_en/pt/'

    # Read from source .txt file and write into .pt file
    for data_split in ['test', 'dev', 'train']:
        path = data_folder + data_split + '.doc.txt'
        raw_data = read_data_from(path, config.tokenizer, ace=args.ace)
        data = remove_irrelevent_data(raw_data[:], args.ere)

        save_to_json(data, save_path + data_split + '.json')
        data_bert_train = pair_trigger_template(data, event_rep, config)
        d_loader = dataset_prepare_trigger_zero_template(data_bert_train, config, trigger_to_ids, metadata)
        save_trigger_dataset(d_loader, save_path + data_split + '.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ace', action='store_true')
    parser.add_argument('--ere', action='store_true')
    parser.add_argument('--data_folder', type=str,  required=True)
    args = parser.parse_args()
    assert args.ere or args.ace is True, 'set either ACE or ERE with --ace or --ere'
    read_from_source(args)


