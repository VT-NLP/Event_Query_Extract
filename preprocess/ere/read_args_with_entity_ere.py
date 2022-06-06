# coding=utf-8
import pandas as pd
from os import listdir
from os.path import isfile, join
import re
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer as spacy_tokenizer
import collections
from collections import defaultdict

spacy_tagger = spacy.load("en_core_web_sm")
spacy_tagger.tokenizer = spacy_tokenizer(spacy_tagger.vocab)
# spacy_tagger.tokenizer.rules = {key: value for key, value in spacy_tagger.tokenizer.rules.items() if "'" not in key and "’" not in key and "‘" not in key}

sents = defaultdict(list)
all_instances = defaultdict(int)
is_trigger_instances = defaultdict(list)
ids = defaultdict(list)
suppliment_trigger = { 'Movement:Transport': 'arrival-travels-penetrated-expelled',
                                    'Conflict:Attack': 'invaded-airstrikes-overthrew-ambused',
                                    'Personnel:End-Position': 'resigning-retired-signed',
                                    'Life:Die': 'deceased-extermination',
                                    'Contact:Meet': 'reunited-retreats',
                                    'Transaction:Transfer-Ownership': 'purchased',
                                    'Personnel:Elect': 'reelected',
                                    'Personnel:Start-Position': 'hiring-rehired-recruited',
                                    'Transaction:Transfer-Money': 'donations-reimbursing-deductions',
                                    'Justice:Sue': 'lawsuits',
                                    'Justice:Execute': 'put to death',
                                    'Life:Injure': 'hopitalised-paralyzed-dismember',
                                    'Contact:Phone-Write': 'emailed-letters',
                                    'Business:Declare-Bankruptcy': 'bankruptcy-bankruptcies-bankrupting-bankrupcy',
                                    'Justice:Appeal': 'appeal',
                                    'Business:End-Org': 'dissolving-disbanded',
                                    'Justice:Arrest-Jail': 'arrested-locked up',
                                    'Justice:Convict': 'pled guilty-convicting',
                                    'Justice:Trial-Hearing': 'trial\'s-hearings',
                                    'Justice:Sentence': 'sentenced',
                                    'Justice:Charge-Indict': 'indictment',
                                    'Conflict:Demonstrate': 'demonstrations',
                                    'Personnel:Nominate': 'nominate',
                                    'Life:Marry': 'married-marriage-marry',
                                    'Business:Start-Org': 'founded',
                                    'Life:Be-Born': 'childbirth',
                                    'Justice:Release-Parole': 'parole',
                                    'Justice:Fine': 'payouts',
                                    'Justice:Pardon': 'pardoned',
                                    'Business:Merge-Org': 'merging-merger',
                                    'Life:Divorce': 'divorce',
                                    'Justice:Acquit': 'acquitted',
                                    'Justice:Extradite': 'extradition',
                                    }
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

for k in suppliment_trigger.keys():
    suppliment_trigger[k] = suppliment_trigger[k].lower().split('-')

count_f = [0]
trigger_c = [0]
arg_c = [0]
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

    arg_set = set()
    entity_set = set()
    pos_set = set()
    trigger_set = set()
    for i in range(len(data)):
        # read triggers

        _sentence, _token_offsets, _event_type, _trigger_idx, _arguments, \
                    _ner_offset, _ner_type, _filler_offset, _filler_type = data[i]
        this_triggers = read_triggers(_token_offsets, _trigger_idx, _event_type, verbose=False)

        # read arguments
        this_arguments = read_arguments(_sentence, _token_offsets, _arguments, _trigger_idx)

        # read entities
        this_entities = read_entities(_token_offsets,_ner_offset, _ner_type)

        this_filler = read_fillers(_sentence, _token_offsets, _filler_offset, _filler_type)
        # add POS tag
        fillers = sum([x[1] for x in this_arguments], [])
        count_f[0] += len([1 for x in fillers if x[0] == 'B'])
        this_pos = get_pos(_sentence)
        if this_triggers:
            trigger_pos_pair1.append([min(np.asarray(this_triggers)[:,i]) for i in range(len(np.asarray(this_triggers)[0]))])
            trigger_pos_pair2.append(this_pos)

        # filter duplicate events
        N = len(this_triggers)
        assert len(this_arguments) == N, 'arguments count not match trigger count'
        keep_ids = []
        event_set = set()
        for i in range(N):
            this_tuple = (' '.join(this_triggers[i]), ' '.join(this_arguments[i][0]), ' '.join(this_arguments[i][1]))
            if this_tuple not in event_set:
                keep_ids.append(i)
        this_triggers = [this_triggers[i] for i in keep_ids]
        this_arguments = sum([this_arguments[i] for i in keep_ids],[])
        assert len(this_arguments) == 2*N, 'arguments count not match trigger count'

        output.append(sum([[list(_sentence)], this_triggers, this_arguments,[], [this_entities], [this_filler], [this_pos]], []))
        # trigger_set = trigger_set | set(sum(this_triggers, []))
        # arg_set = arg_set | set(sum(this_arguments, []))
        # entity_set = entity_set | set(this_entities)
        # pos_set = pos_set | set(this_pos)
    # print("'{}': {},".format(ace_path.split('/')[-1], trigger_c))
    return output, trigger_set, arg_set, entity_set, pos_set


def get_pos(sentence):
    doc = spacy_tagger(' '.join(list(sentence)))
    ret = []
    for token in doc:
        ret.append(token.pos_)
    return ret


def read_entities(token_offsets, ner_offset, ner_type):
    n = len(token_offsets)

    if len(ner_offset) < n:
        ner_offset = np.concatenate((ner_offset, ['O'] * (n - len(ner_offset))))
    if len(ner_type) < n:
        ner_type = np.concatenate((ner_type, ['O'] * (n - len(ner_type))))

    ner = _processed_entities(token_offsets, ner_offset, ner_type)
    return list(ner)


def read_fillers(sentence, token_offsets, filler_offset, filler_type):
    n = len(token_offsets)

    if len(filler_offset) < n:
        filler_offset = np.concatenate((filler_offset, ['O'] * (n - len(filler_offset))))
    if len(filler_type) < n:
        filler_type = np.concatenate((filler_type, ['O'] * (n - len(filler_type))))

    filler = read_entities_from_value(sentence, token_offsets, filler_offset, filler_type)
    return list(filler)


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

            begin_at_in_char = max(this_trigger_begin_at, min_id)
            end_at_in_char = min(this_trigger_end_at, max_id)
            # fix begin and end index
            j = 0
            while begin_at_in_char not in offsets_dic and j < 5:
                begin_at_in_char += 1
                j +=1
            if j == 5: continue  # get rid of this span if the offset is off by too much
            j = 0
            while end_at_in_char not in offsets_dic and j < 5:
                end_at_in_char -= 1
                j += 1
            if j == 5 or begin_at_in_char > end_at_in_char: continue  # skip if begin>end
            try:
                begin_idx = offsets_dic[begin_at_in_char]
                end_idx = offsets_dic[end_at_in_char]
            except:
                print()
            begin_head = get_head_word(sentence, begin_idx, end_idx)
            try:
                entity_bio[begin_head] = 'B-' + this_trigger
            except:
                print()
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
        output = np.where(sentence==ret)[0][0]
    except:
        for i in range(len(ret)):
            if ret in sentence[i]:
                output = i
                break
    return output


def _processed_entities(token_offsets, entity_offset, entity_type):
    n = len(token_offsets)
    ret = read_entities_(token_offsets, entity_offset, entity_type)

    entities = set(entity_offset)
    _idx_begin = [x.split(':')[0] for x in entities if x != 'O']
    _idx_end = [x.split(':')[1] for x in entities if x != 'O']
    if not ret: ret = [['O'] * n]
    ret = np.array([min(np.array(ret)[:,i]) for i in range(n)])
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
    args_all = arguments_info.split('#@#')

    ret = []
    for args in args_all:
        args = re.split('[ :]', args)
        _n = len(args)
        _arguments, _idx_begin, _idx_end = args[1:_n:4], args[2:_n:4], args[3:_n:4]
        _idx = [':'.join([x, y]) for x, y in zip(_idx_begin, _idx_end)]

        arguments_bio = read_args(offsets, _idx, _arguments)
        arguments_bio_value = read_args_filler(sent, offsets, _idx, _arguments,
                                               time_value={'position', 'sentence', 'crime', 'money', 'time'})
        ret.append([arguments_bio, arguments_bio_value])
    return ret


arg_list = []
def read_args(offsets, arg_idxs, arg_type, verbose=False, arg21_ere=True):
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

    arg_list.append(list(set(arg_type)))
    while i < arg_num:
        this_arg_begin_at, this_arg_end_at = list(map(int, arg_idxs[i].split(':')))

        this_arg = arg_type[i]
        if arg21_ere and this_arg in {'time', 'position', 'sentence', 'O', 'crime', 'money'}:
            i += 1
            continue

        # fix begin index
        if this_arg_begin_at in offsets_begin_dic:
            arg_bio[offsets_begin_dic[this_arg_begin_at]] = 'B-' + this_arg
            begin_idx_fixed = offsets_begin_dic[this_arg_begin_at]
            arg_c[0] += 1
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
            arg_bio[begin_idx_fixed] = 'B-' + this_arg
            arg_c[0] += 1

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
            arg_bio[begin_idx_fixed+1:end_idx_fixed+1] = ['I-' + this_arg] * (end_idx_fixed - begin_idx_fixed)

        i += 1

    return arg_bio


def read_args_filler(sent, offsets, arg_idxs, arg_type, time_value=None, verbose=False):
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
    arg_bio_filler = ['O'] * len(offsets)
    i = 0
    cont = False
    while i < arg_num:
        this_arg_begin_at, this_arg_end_at = list(map(int, arg_idxs[i].split(':')))

        this_arg = arg_type[i]
        if this_arg not in time_value:
            i += 1
            continue

        # fix begin index
        if this_arg_begin_at in offsets_begin_dic:
            if arg_bio_filler[offsets_begin_dic[this_arg_begin_at]] != 'O':
                arg_bio_filler[offsets_begin_dic[this_arg_begin_at]] += '#@#B-' + this_arg
            else:
                arg_bio_filler[offsets_begin_dic[this_arg_begin_at]] = 'B-' + this_arg
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
            if arg_bio_filler[begin_idx_fixed] != 'O':
                arg_bio_filler[begin_idx_fixed] = '#@#B-' + this_arg
            else:
                arg_bio_filler[begin_idx_fixed] = 'B-' + this_arg

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

        head_idx = get_head_word(sent, begin_idx_fixed, end_idx_fixed)
        arg_bio_filler[head_idx] = 'B-' + this_arg

        i += 1

    return arg_bio_filler

#
# def read_args_filler(offsets, arg_idxs, arg_type, verbose=False, arg22_ace=True, time_value=None):
#     """
#     Get args and write a row for each arg
#     :param offsets: token offsets
#     :param arg_idxs: triggers' indices
#     :param arg_type: triggers' types
#     :param verbose:
#     :return: [[BIO for trigger1], [BIO for trigger2], ...]
#     """
#     offsets_begin_dic, offsets_end_dic = offset2dic(offsets)
#     arg_num = len(arg_idxs)
#     arg_bio_time = ['O'] * len(offsets)
#     arg_bio_value = ['O'] * len(offsets)
#     i = 0
#     cont = False
#     while i < arg_num:
#         this_arg_begin_at, this_arg_end_at = list(map(int, arg_idxs[i].split(':')))
#
#         this_arg = arg_type[i]
#         if arg22_ace and this_arg not in time_value:
#             i += 1
#             continue
#
#         # fix begin index
#         if this_arg_begin_at in offsets_begin_dic:
#             if arg_bio_value[offsets_begin_dic[this_arg_begin_at]] != 'O':
#                 arg_bio_value[offsets_begin_dic[this_arg_begin_at]] += '#@#B-' + this_arg
#             else:
#                 arg_bio_value[offsets_begin_dic[this_arg_begin_at]] = 'B-' + this_arg
#             begin_idx_fixed = offsets_begin_dic[this_arg_begin_at]
#         else:
#             if verbose: print('fix begin index')
#             while this_arg_begin_at not in offsets_begin_dic:
#                 this_arg_begin_at -= 1
#                 if this_arg_begin_at < min(offsets_begin_dic.keys()) or this_arg_end_at > max(offsets_end_dic.keys()):
#                     cont = True
#                     break
#             if cont:
#                 cont = False
#                 i += 1
#                 continue
#             begin_idx_fixed = offsets_begin_dic[this_arg_begin_at]
#             if arg_bio_value[begin_idx_fixed] != 'O':
#                 arg_bio_value[begin_idx_fixed] = '#@#B-' + this_arg
#             else:
#                 arg_bio_value[begin_idx_fixed] = 'B-' + this_arg
#
#         # fix end index
#         if this_arg_end_at in offsets_end_dic:
#             end_idx_fixed = offsets_end_dic[this_arg_end_at]
#         else:
#             if verbose: print('fix end index')
#             while this_arg_end_at not in offsets_end_dic:
#                 this_arg_end_at += 1
#                 if this_arg_end_at > max(offsets_end_dic.keys()):
#                     cont = True
#                     break
#             if cont:
#                 i += 1
#                 cont = False
#                 continue
#             end_idx_fixed = offsets_end_dic[this_arg_end_at]
#
#         # add I- prefix
#         if begin_idx_fixed < end_idx_fixed:
#             for idx in range(begin_idx_fixed+1, end_idx_fixed+1):
#                 if arg_bio_value[idx] == 'O':
#                     arg_bio_value[idx] = 'I-' + this_arg
#                 else:
#                     arg_bio_value[idx] += '#@#I-' + this_arg
#             trigger_c[0] += 1
#
#         i += 1
#
#     return arg_bio_time, arg_bio_value


def extract_data(df_ace):
    """
    Read the ACE2005 csv files into a list of data tuples
    :param df_ace: ACE data frame
    :return: a list of data tuples [(sentence, token_offsets, event_type, trigger_idx, arguments)]
    """
    sent_sep = '----sentence_delimiter----'

    tokens = df_ace['token'].values
    token_offsets = df_ace['offset'].values
    arguments = df_ace['trigger_arguments'].values
    trigger_idxs = df_ace['trigger_offset'].values
    trigger_type = df_ace['trigger_type'].values
    ner_offset = df_ace['ner_offset'].values
    ner_type = df_ace['ner_type'].values
    filler_type = df_ace['filler_type'].values
    filler_offset = df_ace['filler_offset'].values

    tok_num = len(tokens)
    dataset = []
    begin = 0

    for i in range(tok_num):
        if tokens[i] == sent_sep:
            data = (tokens[begin:i], token_offsets[begin:i],
                    trigger_type[begin:i], trigger_idxs[begin:i],
                    arguments[begin:i],
                    ner_offset[begin:i], ner_type[begin:i],
                    filler_offset[begin:i], filler_type[begin:i])
            dataset.append(data)
            begin = i + 1
        elif i == tok_num-1:
            data = (tokens[begin:], token_offsets[begin:],
                    trigger_type[begin:], trigger_idxs[begin:],
                    arguments[begin:],
                    ner_offset[begin:], ner_type[begin:],
                    filler_offset[begin:i], filler_type[begin:i])
            dataset.append(data)
            begin = i + 1

    return dataset


def read_triggers(offsets, trigger_idxs, trigger_type, verbose=False):
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

    seen_event = set()
    while i < tok_num:
        if trigger_idxs[i] == 'O':
            i += 1
            continue
        trigger_id_list = trigger_idxs[i].split('#@#')
        trigger_type_list = trigger_type[i].split('#@#')
        assert len(trigger_id_list) == len(trigger_type_list), 'Trigger indices count does not match with event count'
        if trigger_idxs[i] in seen_event:
            i += 1
            continue
        seen_event.add(trigger_idxs[i])

        for this_trigger_type, this_trigger_idxs in zip(trigger_type_list, trigger_id_list):
            this_trigger_begin_at, this_trigger_end_at = list(map(int, this_trigger_idxs.split(':')))

            # fix begin index
            if this_trigger_begin_at in offsets_begin_dic:
                trigger_bio[offsets_begin_dic[this_trigger_begin_at]] = 'B-' + this_trigger_type
                begin_idx_fixed = offsets_begin_dic[this_trigger_begin_at]
            else:
                if verbose:print('fix begin index')
                while this_trigger_begin_at not in offsets_begin_dic:
                    this_trigger_begin_at -= 1
                    if this_trigger_begin_at < min(offsets_begin_dic.keys()) or this_trigger_end_at > max(offsets_end_dic.keys()):
                        cont = True
                        break
                if cont:
                    cont = False
                    # i += 1
                    continue
                begin_idx_fixed = offsets_begin_dic[this_trigger_begin_at]
                trigger_bio[begin_idx_fixed] = 'B-' + this_trigger_type

            # fix end index
            if this_trigger_end_at in offsets_end_dic:
                end_idx_fixed = offsets_end_dic[this_trigger_end_at]
            else:
                if verbose:print('fix end index')
                while this_trigger_end_at not in offsets_end_dic:
                    this_trigger_end_at += 1
                    if this_trigger_end_at > max(offsets_end_dic.keys()):
                        cont = True
                        break
                if cont:
                    # i += 1
                    cont = False
                    continue
                end_idx_fixed = offsets_end_dic[this_trigger_end_at]

            # add I- prefix
            if begin_idx_fixed < end_idx_fixed:
                trigger_bio[begin_idx_fixed+1:end_idx_fixed+1] = ['I-' + this_trigger_type] * (end_idx_fixed - begin_idx_fixed)

            # i = max(begin_idx_fixed, end_idx_fixed)
            output.append(trigger_bio)
            # Next trigger
            trigger_bio = ['O'] * tok_num
        i += 1

    return output


def read_entities_(offsets, trigger_idxs, trigger_type, verbose=False):
    """
    Get entities and write a row for each trigger
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

    seen_event = set()
    while i < tok_num:
        if trigger_idxs[i] == 'O':
            i += 1
            continue
        trigger_idxs[i] = trigger_idxs[i].split('#')[0]

        this_trigger_begin_at, this_trigger_end_at = list(map(int, trigger_idxs[i].split(':')))
        this_trigger = trigger_type[i]

        seen_event.add((this_trigger_begin_at, this_trigger_end_at))
        trigger_c[0] += len(set(this_trigger.split('#@#')))

        # fix begin index
        if this_trigger_begin_at in offsets_begin_dic:
            trigger_bio[offsets_begin_dic[this_trigger_begin_at]] = 'B-' + this_trigger
            begin_idx_fixed = offsets_begin_dic[this_trigger_begin_at]
        else:
            if verbose:print('fix begin index')
            while this_trigger_begin_at not in offsets_begin_dic:
                this_trigger_begin_at -= 1
                if this_trigger_begin_at < min(offsets_begin_dic.keys()) or this_trigger_end_at > max(offsets_end_dic.keys()):
                    cont = True
                    break
            if cont:
                cont = False
                i += 1
                continue
            begin_idx_fixed = offsets_begin_dic[this_trigger_begin_at]
            trigger_bio[begin_idx_fixed] = 'B-' + this_trigger

        # fix end index
        if this_trigger_end_at in offsets_end_dic:
            end_idx_fixed = offsets_end_dic[this_trigger_end_at]
        else:
            if verbose:print('fix end index')
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
            trigger_bio[begin_idx_fixed+1:end_idx_fixed+1] = ['I-' + this_trigger] * (end_idx_fixed - begin_idx_fixed)

        i = max(begin_idx_fixed, end_idx_fixed)
        i += 1
        output.append(trigger_bio)
        # Next trigger
        trigger_bio = ['O'] * tok_num
    # if output:
    #     try:
    #         for i in range(len(output[0])):
    #             this =[x[i] for x in output]
    #             this = set(this)
    #             if 'O' in this:
    #                 this.remove('O')
    #             if len(this) > 1:
    #                 pass
    #     except:
    #         print()
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
        for j in range(offsets_begin[i], offsets_end[i]+1):
            offsets_dic[j] = i
    return offsets_dic


def read_ace_dir_to_bio(ace_files, output_path, verbose=False, trigger_set__=None):
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
        data, triggers, args, entities, poses = read_arguments_from_ace(f)
        for i in range(len(data)):
            data[i].append(f.split('/')[-1])
        output.extend(data)

    f = open(output_path, 'w')
    for i in range(len(output)):
        f.writelines('\n'.join([' '.join(i) for i in output[i][:-1]]))
        f.writelines('\n' + output[i][-1])
        f.writelines('\n\n')
    f.close()
    return 0


def read_ace_with_entity(ace_dir, output_dir, split_path, trigger_set=None):
    """
    Extrace ace2005 with train_test_split files, and write into a single file for each dataset.
    This file also includes entity information
    :param ace_dir: the path to ace2005 data
    :param output_dir: the output path
    :param split_path: path to split files
    :return: None
    """
    splits = ['train.doc.txt', 'dev.doc.txt','test.doc.txt']
    splits = ['test.doc.txt']
    # set_trigger = set()
    # triggers = []
    for data in splits:
        print('\n===================================================================')
        print('Processing ', data[:-4])
        output_path = ''.join([output_dir, data])
        files = open(split_path + data, 'r').read().splitlines()
        files = [x[:-4] if 'kbp' in x else x for x in files]
        files = [''.join([ace_dir, i.split('/')[-1], '.csv']) for i in files]
        read_ace_dir_to_bio(files, output_path, trigger_set__=trigger_set)

    return 0


if __name__ == '__main__':
    path = './code/data/ere_en/all_ere/'
    split_path = './code/oneIE_v0.4.3 copy/resource/splits/ERE-EN/'
    output = './code/data/ere_en/train_files_with_value_time/head'
    read_ace_with_entity(path, output, split_path)
