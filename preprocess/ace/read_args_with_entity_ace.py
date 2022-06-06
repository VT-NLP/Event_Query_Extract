# coding=utf-8
import pandas as pd
import os
from os.path import isfile, join
import re
import numpy as np
import argparse
import spacy
from spacy.tokenizer import Tokenizer as spacy_tokenizer
import collections
from collections import defaultdict

spacy_tagger = spacy.load("en_core_web_sm")
spacy_tagger.tokenizer = spacy_tokenizer(spacy_tagger.vocab)

global_tmp = {"alt.gossip.celebrities_20041118.2331.csv",
              "rec.arts.mystery_20050219.1126.csv",
              "rec.sport.disc_20050209.2202.csv",
              "seattle.politics_20050122.2412.csv",
              "soc.culture.hmong_20050210.1130.csv",
              "soc.culture.jewish_20050130.2105.csv",
              "MARKBACKER_20041119.1002.csv",
              "MARKBACKER_20041128.1641.csv",
              "MARKBACKER_20050103.0829.csv",
              "OIADVANTAGE_20041224.1007.csv"}

trigger_pos_pair1 = []
trigger_pos_pair2 = []
trigger_c = [0]


def read_arguments_from_ace(ace_path, SEN_SEP='SENT_SEP'):
    """
    Read arguments from ace2005
    :param ace_path:
    :param SEN_SEP:
    :return: a list of extracted data, [[tokens, trigger1_BIO, ..., triggerk_BIO, arg1_BIO, ..., argk_BIO], ...]
    """
    # read ace file as dataframe

    if ace_path.split('/')[-1] in global_tmp:
        pass
    df_ace = pd.read_csv(ace_path)
    df_ace = df_ace.fillna(SEN_SEP)
    data = extract_data(df_ace)

    output = []
    # trigger_c = 0
    for i in range(len(data)):
        # read triggers
        _sentence, _token_offsets, _event_type, _trigger_idx, _arguments, \
        _ner_offset, _ner_type, \
        _time_offset, _value_offset, _value_type = data[i]
        this_triggers = read_triggers(_token_offsets, _trigger_idx, _event_type, is_trigger=True, verbose=True)

        # read arguments
        this_arguments = read_arguments(_sentence, _token_offsets, _arguments, _trigger_idx)

        # read entities
        ner = processed_entities(_token_offsets, _ner_offset, _ner_type)

        output.append(sum([[list(_sentence)], this_triggers, this_arguments, [ner]], []))
    return output


def read_entities_from_value(sentence, offsets, entity_idxs, entity_type):
    """
    Get triggers and write a row for each trigger
    :param offsets: token offsets
    :param entity_idxs: entities' indices
    :param entity_type: entities' types
    :param verbose:
    :return: [[BIO for trigger1], [BIO for trigger2], ...]
    """
    offsets_dic = offset2dic_v2(offsets)
    min_id = min(offsets_dic.keys())
    max_id = max(offsets_dic.keys())
    tok_num = len(entity_idxs)
    entity_bio = ['O'] * len(offsets)
    # add 'B-'/'I-' for beginning/inside of trigger span
    i = 0
    while i < tok_num:
        # pass if 'O'
        if entity_idxs[i] == 'O':
            i += 1
            continue

        tmp_idx = entity_idxs[i].split('#@#')
        tmp_entity = entity_type[i].split('#@#')
        for j in range(len(tmp_idx)):
            this_trigger_begin_at, this_trigger_end_at = list(map(int, tmp_idx[j].split(':')))
            try:
                this_trigger = tmp_entity[j]
            except:
                this_trigger = 'TIME'
            begin_idx = offsets_dic[max(this_trigger_begin_at, min_id)]
            end_idx = offsets_dic[min(this_trigger_end_at, max_id)]

            begin_head = get_head_word(sentence, begin_idx, end_idx)

            if 'Contact' in this_trigger:
                continue
            entity_bio[begin_head] = 'B-' + this_trigger
        i += 1

    return entity_bio


def get_head_word(sentence, begin_idx, end_idx):
    doc = spacy_tagger(' '.join(sentence[begin_idx:end_idx + 1]))
    ret = sentence[begin_idx]
    output = begin_idx
    for token in doc:
        if token.dep_ == 'ROOT':
            ret = token.text
            break
    try:
        output = np.where(sentence == ret)[0][0]
    except:
        for i in range(len(ret)):
            if ret in sentence[i]:
                output = i
                break
    return output


def processed_entities(token_offsets, entity_offset, entity_type):
    n = len(token_offsets)
    ret = read_triggers(token_offsets, entity_offset, entity_type, is_entity=True)

    entities = set(entity_offset)
    _idx_begin = [x.split(':')[0] for x in entities if x != 'O']
    _idx_end = [x.split(':')[1] for x in entities if x != 'O']
    if not ret: ret = [['O'] * n]
    ret = [min(np.array(ret)[:, i]) for i in range(n)]
    return ret


def read_arguments(sent, offsets, arguments, trigger_idx):
    set_ = set()
    _arguments = []
    for i in range(len(trigger_idx)):
        # avoid add duplicates for the same event
        if trigger_idx[i] == 'O' or trigger_idx[i] in set_:
            continue
        set_.add(trigger_idx[i])
        _arguments.append(arguments[i])
    tok_num = len(_arguments)
    arguments_bio = []
    for i in range(tok_num):
        arguments_bio.extend(gen_arguments_bio(sent, offsets, _arguments[i]))
    return arguments_bio


def gen_arguments_bio(sent, offsets, arguments_info):
    args = re.split('[ :]', arguments_info)
    _n = len(args)
    _arguments, _idx_begin, _idx_end = args[1:_n:4], args[2:_n:4], args[3:_n:4]
    _idx = [':'.join([x, y]) for x, y in zip(_idx_begin, _idx_end)]
    arguments_bio = read_args(offsets, _idx, _arguments)
    arguments_bio_value = read_args_value_time(sent, offsets, _idx, _arguments,
                                               time_value={'Position', 'Sentence', 'Crime', 'Money', 'Price'})
    arguments_bio_time = read_args_value_time(sent, offsets, _idx, _arguments,
                                              time_value={'Time-At-End', 'Time-Within', 'Time-Holds',
                                                          'Time-Starting', 'Time-Before', 'Time-At-Beginning',
                                                          'Time-After', 'Time-Ending'}, time=True)
    return [arguments_bio, arguments_bio_value, arguments_bio_time]


def read_args(offsets, arg_idxs, arg_type, verbose=False, arg22_ace=True):
    """
    Get args and write a row for each arg
    :param offsets: token offsets
    :param arg_idxs: triggers' indices
    :param arg_type: triggers' types
    :param verbose:
    :return: [[BIO for trigger1], [BIO for trigger2], ...]
    """
    offsets_begin_dic, offsets_end_dic = offset2dic(offsets)
    arg_num = len(arg_idxs)
    arg_bio = ['O'] * len(offsets)
    i = 0
    cont = False
    while i < arg_num:
        this_arg_begin_at, this_arg_end_at = list(map(int, arg_idxs[i].split(':')))

        this_arg = arg_type[i]
        if arg22_ace and this_arg in {
            'Time-At-End', 'Time-Within', 'Time-Holds', 'Time-Starting', 'Time-Before',
            'Time-At-Beginning', 'Time-After', 'Time-Ending',
            'Position', 'Sentence', 'Crime', 'Money', 'Price'}:
            i += 1
            continue

        # fix begin index
        if this_arg_begin_at in offsets_begin_dic:

            if arg_bio[offsets_begin_dic[this_arg_begin_at]] != 'O':
                arg_bio[offsets_begin_dic[this_arg_begin_at]] += '#@#B-' + this_arg
            else:
                arg_bio[offsets_begin_dic[this_arg_begin_at]] = 'B-' + this_arg
            begin_idx_fixed = offsets_begin_dic[this_arg_begin_at]
        else:
            if verbose: print('fix begin index')
            while this_arg_begin_at not in offsets_begin_dic:
                this_arg_begin_at -= 1
                if this_arg_begin_at < min(offsets_begin_dic.keys()) or this_arg_end_at > max(offsets_end_dic.keys()):
                    cont = True
                    break
            if cont:
                cont = False
                i += 1
                continue
            begin_idx_fixed = offsets_begin_dic[this_arg_begin_at]

            if arg_bio[begin_idx_fixed] != 'O':
                arg_bio[begin_idx_fixed] = '#@#B-' + this_arg
            else:
                arg_bio[begin_idx_fixed] = 'B-' + this_arg

        # fix end index
        if this_arg_end_at in offsets_end_dic:
            end_idx_fixed = offsets_end_dic[this_arg_end_at]
        else:
            if verbose: print('fix end index')
            while this_arg_end_at not in offsets_end_dic:
                this_arg_end_at += 1
                if this_arg_end_at > max(offsets_end_dic.keys()):
                    cont = True
                    break
            if cont:
                i += 1
                cont = False
                continue
            end_idx_fixed = offsets_end_dic[this_arg_end_at]

        # add I- prefix
        if begin_idx_fixed < end_idx_fixed:
            for idx in range(begin_idx_fixed + 1, end_idx_fixed + 1):

                if arg_bio[idx] == 'O':
                    arg_bio[idx] = 'I-' + this_arg
                else:
                    arg_bio[idx] += '#@#I-' + this_arg
            trigger_c[0] += 1

        i += 1

    return arg_bio


def read_args_value_time(sent, offsets, arg_idxs, arg_type, verbose=False, arg22_ace=True, time_value=None, time=False):
    """
    Get args and write a row for each arg
    :param offsets: token offsets
    :param arg_idxs: triggers' indices
    :param arg_type: triggers' types
    :param verbose:
    :return: [[BIO for trigger1], [BIO for trigger2], ...]
    """
    offsets_begin_dic, offsets_end_dic = offset2dic(offsets)
    arg_num = len(arg_idxs)
    arg_bio = ['O'] * len(offsets)
    i = 0
    cont = False
    while i < arg_num:
        this_arg_begin_at, this_arg_end_at = list(map(int, arg_idxs[i].split(':')))

        this_arg = arg_type[i]
        if arg22_ace and this_arg not in time_value:
            i += 1
            continue

        # fix begin index
        if this_arg_begin_at in offsets_begin_dic:
            begin_idx_fixed = offsets_begin_dic[this_arg_begin_at]
        else:
            if verbose: print('fix begin index')
            while this_arg_begin_at not in offsets_begin_dic:
                this_arg_begin_at -= 1
                if this_arg_begin_at < min(offsets_begin_dic.keys()) or this_arg_end_at > max(offsets_end_dic.keys()):
                    cont = True
                    break
            if cont:
                cont = False
                i += 1
                continue
            begin_idx_fixed = offsets_begin_dic[this_arg_begin_at]

        # fix end index
        if this_arg_end_at in offsets_end_dic:
            end_idx_fixed = offsets_end_dic[this_arg_end_at]
        else:
            if verbose: print('fix end index')
            while this_arg_end_at not in offsets_end_dic:
                this_arg_end_at += 1
                if this_arg_end_at > max(offsets_end_dic.keys()):
                    cont = True
                    break
            if cont:
                i += 1
                cont = False
                continue
            end_idx_fixed = offsets_end_dic[this_arg_end_at]

        # get head word of the argument span and assign argument label to the head word
        head_idx = get_head_word(sent, begin_idx_fixed, end_idx_fixed)
        if time:
            arg_bio[head_idx] = 'B-Time'
        else:
            arg_bio[head_idx] = 'B-' + this_arg
        i += 1

    return arg_bio


def extract_data(df_ace):
    """
    Read the ACE2005 csv files into a list of data tuples
    :param df_ace: ACE data frame
    :return: a list of data tuples [(sentence, token_offsets, event_type, trigger_idx, arguments)]
    """
    global_tmp = {"alt.gossip.celebrities_20041118.2331.csv",
                  "rec.arts.mystery_20050219.1126.csv",
                  "rec.sport.disc_20050209.2202.csv",
                  "seattle.politics_20050122.2412.csv",
                  "soc.culture.hmong_20050210.1130.csv",
                  "soc.culture.jewish_20050130.2105.csv",
                  "MARKBACKER_20041119.1002.csv",
                  "MARKBACKER_20041128.1641.csv",
                  "MARKBACKER_20050103.0829.csv",
                  "OIADVANTAGE_20041224.1007.csv"}

    sent_sep = '----sentence_delimiter----'

    tokens = df_ace['token'].values
    token_offsets = df_ace['offset'].values
    arguments = df_ace['trigger_arguments'].values
    trigger_idxs = df_ace['trigger_offset'].values
    trigger_type = df_ace['trigger_type'].values
    ner_offset = df_ace['ner_offset'].values
    ner_type = df_ace['ner_type'].values
    time_offset = df_ace['timex2_offset'].values
    value_offset = df_ace['value_offset'].values
    value_type = df_ace['value_type'].values

    tok_num = len(tokens)

    dataset = []
    begin = 0

    if token_offsets[0].split(':')[0] + '.csv' in global_tmp:
        pass
    for i in range(tok_num):
        if tokens[i] == sent_sep:
            data = (tokens[begin:i], token_offsets[begin:i],
                    trigger_type[begin:i], trigger_idxs[begin:i],
                    arguments[begin:i],
                    ner_offset[begin:i], ner_type[begin:i],
                    time_offset[begin:i], value_offset[begin:i], value_type[begin:i])
            dataset.append(data)
            begin = i + 1
        elif i == tok_num - 1:
            data = (tokens[begin:], token_offsets[begin:],
                    trigger_type[begin:], trigger_idxs[begin:],
                    arguments[begin:],
                    ner_offset[begin:], ner_type[begin:],
                    time_offset[begin:], value_offset[begin:], value_type[begin:])
            dataset.append(data)
            begin = i + 1

    return dataset


def read_triggers(offsets, trigger_idxs, trigger_type, verbose=False, is_arguments=False, is_trigger=False,
                  is_entity=False):
    """
    Get triggers and write a row for each trigger
    :param offsets: token offsets
    :param trigger_idxs: triggers' indices
    :param trigger_type: triggers' types
    :param verbose:
    :return: [[BIO for trigger1], [BIO for trigger2], ...]
    """
    offsets_begin_dic, offsets_end_dic = offset2dic(offsets)
    output = []
    tok_num = len(trigger_idxs)
    trigger_bio = ['O'] * len(offsets)
    # add 'B-'/'I-' for beginning/inside of trigger span
    i = 0
    cont = False

    while i < tok_num:
        # pass if 'O'
        if not is_arguments and trigger_idxs[i] == 'O':
            i += 1
            continue

        trigger_idxs[i] = trigger_idxs[i].split('#@#')[0]  # confirmed that there is no overlap spans
        this_trigger_begin_at, this_trigger_end_at = list(map(int, trigger_idxs[i].split(':')))
        this_trigger = trigger_type[i]

        # fix begin index
        if this_trigger_begin_at in offsets_begin_dic:
            trigger_bio[offsets_begin_dic[this_trigger_begin_at]] = 'B-' + this_trigger
            begin_idx_fixed = offsets_begin_dic[this_trigger_begin_at]
        else:
            if verbose: print('fix begin index')
            while this_trigger_begin_at not in offsets_begin_dic:
                this_trigger_begin_at -= 1
                if this_trigger_begin_at < min(offsets_begin_dic.keys()) or this_trigger_end_at > max(
                        offsets_end_dic.keys()):
                    cont = True
                    break
            if cont:
                cont = False
                i += 1
                continue
            begin_idx_fixed = offsets_begin_dic[this_trigger_begin_at]
            if not is_arguments and trigger_type[begin_idx_fixed] != 'O':
                trigger_bio[begin_idx_fixed] = 'B-' + this_trigger
            else:
                trigger_bio[begin_idx_fixed] = 'B-' + this_trigger

        # fix end index
        if this_trigger_end_at in offsets_end_dic:
            end_idx_fixed = offsets_end_dic[this_trigger_end_at]
        else:
            if verbose: print('fix end index')
            while this_trigger_end_at not in offsets_end_dic:
                this_trigger_end_at += 1
                if this_trigger_end_at > max(offsets_end_dic.keys()):
                    cont = True
                    break
            if cont:
                i += 1
                cont = False
                continue
            end_idx_fixed = offsets_end_dic[this_trigger_end_at]

        # add I- prefix
        if begin_idx_fixed < end_idx_fixed:
            trigger_bio[begin_idx_fixed + 1:end_idx_fixed + 1] = ['I-' + this_trigger] * (
                        end_idx_fixed - begin_idx_fixed)
        if not is_arguments:
            i = max(begin_idx_fixed, end_idx_fixed)
        i += 1

        if not is_arguments:
            # append for each trigger
            output.append(trigger_bio)
            # Next trigger
            trigger_bio = ['O'] * tok_num
    if is_arguments:
        return trigger_bio
    # trigger_count[0] += len(output)
    if output:
        for i in range(len(output[0])):
            this = [x[i] for x in output]
            this = set(this)
            if 'O' in this:
                this.remove('O')
            if len(this) > 1:
                pass
    return output


def offset2dic(offsets):
    """
    Generate two dictionaries for begin indices and end indices
    :param offsets: a list of offsets in ['fileID:begin-end', 'fileID:begin-end', ....]
    :return:
    """
    offsets = [i.split(':')[1] for i in offsets]
    offsets_begin = [int(i.split('-')[0]) for i in offsets]
    offsets_end = [int(i.split('-')[1]) for i in offsets]

    tok_num = len(offsets)  # total number of tokens in current file

    # create two dictionaries to look up tokens indices by offsets
    _range = range(tok_num)
    offsets_begin_dic = dict(zip(offsets_begin, _range))
    offsets_end_dic = dict(zip(offsets_end, _range))
    return offsets_begin_dic, offsets_end_dic


def offset2dic_v2(offsets):
    """
    Generate two dictionaries for begin indices and end indices
    :param offsets: a list of offsets in ['fileID:begin-end', 'fileID:begin-end', ....]
    :return:
    """
    offsets = [i.split(':')[1] for i in offsets]
    offsets_begin = [int(i.split('-')[0]) for i in offsets]
    offsets_end = [int(i.split('-')[1]) for i in offsets]

    tok_num = len(offsets)  # total number of tokens in current file

    # create two dictionaries to look up tokens indices by offsets
    _range = range(tok_num)
    offsets_dic = dict()
    for i in range(tok_num):
        for j in range(offsets_begin[i], offsets_end[i] + 1):
            offsets_dic[j] = i
    return offsets_dic


def read_ace_dir_to_bio(ace_files, output_path):
    """
    read ace2005 data and write into a single extracted file
    :param ace_files:
    :param output_path:
    :param verbose:
    :return:
    """
    files = ace_files
    output = []

    for f in files:
        data = read_arguments_from_ace(f)
        for i in range(len(data)):
            data[i].append(f.split('/')[-1])
        output.extend(data)

    # save extracted data
    f = open(output_path, 'w')
    for i in range(len(output)):
        f.writelines('\n'.join([' '.join(i) for i in output[i][:-1]]))
        f.writelines('\n' + output[i][-1])
        f.writelines('\n\n')
    f.close()
    return 0


def read_ace_with_entity(data_dir, output_dir, split_path):
    """
    Extract annotated data with train_test_split files, and write into a single file for each split.
    This file also includes entity information
    :param data_dir: the path to annotated data
    :param output_dir: the output path
    :param split_path: path to split files
    :return: None
    """
    splits = ['train.doc.txt', 'dev.doc.txt', 'test.doc.txt']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for data in splits:
        print('\n===================================================================')
        print('Processing ', data[:-4])
        output_path = ''.join([output_dir, data])
        files = open(split_path + data, 'r').read().splitlines()
        files = [''.join([data_dir, i.split('/')[-1], '.csv']) for i in files]
        read_ace_dir_to_bio(files, output_path)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ace', action='store_true')
    parser.add_argument('--ere', action='store_true')
    parser.add_argument('--path', default='../data/ace_en/ace/', type=str, required=True)
    parser.add_argument('--split_path', default='../data/splits/ACE05-E/', type=str, required=True)
    parser.add_argument('--output', default='../data/ace_en/preprocessed_data/', type=str, required=True)
    args = parser.parse_args()
    assert args.ere or args.ace is True, 'set either ACE or ERE with --ace or --ere'
    if args.ace:
        read_ace_with_entity(args.path, args.output, args.split_path)
    else:
        pass
