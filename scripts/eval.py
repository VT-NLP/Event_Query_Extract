from model.entity_detection.entity_detection import EntityDetection
from utils.config import Config
import torch
from utils.metadata import Metadata
from model.trigger_detection.trigger_detection import TriggerDetection
from utils.utils_model import pred_to_event_mention
from preprocess.save_dataset import token_to_berttokens, pair_trigger_template, \
    dataset_prepare_trigger_zero_template, get_event_rep, remove_irrelevent_data
from utils.data_to_dataloader import read_data_from, arg_to_token_ids
from utils.utils_model import load_from_jsonl, save_to_jsonl
from tqdm import tqdm
from utils.utils_model import calculate_f1
import numpy as np
import argparse


def extract_spans(x, entity_to_ids):
    # extract entity spans from predictions
    N = len(x)
    i = 0
    begin, end, tmp_tag = -1, -1, None
    spans = set()
    while i < N:
        # if the label is 'O' and no tmp tag stored, continue
        if x[i] == entity_to_ids['O']:
            begin, end = -1, -1
            i += 1
            tmp_tag = None
            continue
        elif x[i] < entity_to_ids['O']:
            if tmp_tag:
                end = i - 1
                spans.add((begin, end+1, tmp_tag.item()))
            begin = i
            end = i
            tmp_tag = x[i]
            while i < N - 1 and x[i] == x[i + 1]:
                i += 1
                end = i
            spans.add((begin, end+1, tmp_tag.item()))
        i += 1
    if tmp_tag:
        spans.add((begin, end+1, tmp_tag.item()))
    return spans


def from_entity_identifier_to_entity_matrix(entity_identifier, max_entity_count=40):
    # transform the entity token indicator to an entity mapping matrix
    N = len(entity_identifier)
    entity_matrix = torch.zeros((N, max_entity_count))

    for i in range(len(entity_identifier)):
        if entity_identifier[i] < 0:
            continue
        else:
            this_entity_span = torch.sum(entity_identifier == entity_identifier[i]).float()
            entity_matrix[i, entity_identifier[i]] = 1. / this_entity_span

    return entity_matrix


def event_id_to_arg_query_and_mask(event_arg_dic, args_to_token_ids, args_to_type_ids):
    """
    For each event, compose an argument query and an argument masking
    :param event_arg_dic:
    :param args_to_token_ids:
    :param args_to_type_ids
    :return: a dictionary of the following structure: {event_type: (arg_query_bert, arg_query_mask, arg_ids)}
    """
    ret = dict()
    max_w = max([len(v) for v in event_arg_dic.values()])
    queries = [sum([args_to_token_ids[arg] for arg in v], []) for v in event_arg_dic.values()]
    queries = pad_sequences(queries, dtype="long", truncating="post", padding="post")
    max_h = queries.shape[1]

    i = 0
    for event in sorted(list(event_arg_dic.keys())):
        begin, end = 0, 0
        j = 0
        this_query = []
        # extra slot for [SEP] to indicate 'O'
        this_mask = np.zeros((max_h+1, max_w+1))
        arg_ids = []
        for arg in sorted(list(event_arg_dic[event])):
            this_query.append(args_to_token_ids[arg])
            end += len(args_to_token_ids[arg])
            this_mask[begin:end, j] = 1 / (end - begin)
            j += 1
            begin = end
            arg_ids.append(args_to_type_ids[arg])
        arg_ids.append(args_to_type_ids['O'])
        this_query = np.array(sum(this_query, []), dtype=int)
        this_mask[end, j] = 1
        ret[i] = (this_query, this_mask, arg_ids)
        i += 1
    return ret


from keras.preprocessing.sequence import pad_sequences
def pad_seq(data, pad_value=0, dtype='long'):
    N = len(data)
    for i in range(N):
        data[i] = np.array(data[i])
    maxlen = max([len(x) for x in data])
    data = pad_sequences(data, maxlen=maxlen, dtype=dtype, truncating="post", padding="post", value=pad_value)
    return torch.Tensor(data).cuda()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_ere', default=True, type=bool)
    parser.add_argument('--save_arg_pt', default=False, type=bool)
    parser.add_argument('--trigger_threshold', default=1.0)
    parser.add_argument('--arg_threshold', default=0.)
    parser.add_argument('--data_split', default='dev')
    args = parser.parse_args()

    # config
    metadata = Metadata()
    config = Config()
    config.pretrained_weights = 'bert'
    config.ere = False
    config.trigger_threshold=args.trigger_threshold
    config.arg_threshold=args.arg_threshold

    # load data
    data_split = args.data_split
    if config.ere:
        config.event_count = 38
        trigger_to_ids = metadata.ere.triggers_to_ids
        path = 'data/ere_en/processed_data/' + data_split + '.doc.txt'
        raw_data = read_data_from(path, config.tokenizer, ace=not config.ere)
        data = remove_irrelevent_data(raw_data[:], config.ere)
        gth_annotations = load_from_jsonl('data/ere_en/pt/'+ data_split +'.json')[0]
        meta_info = metadata.ere
        event_rep = get_event_rep(config.project_root + '/preprocess/ere/trigger_representation.json', 'event_name_seed')
    else:
        config.event_count = 33
        trigger_to_ids = metadata.ace.triggers_to_ids
        path = 'data/ace_en/processed_data/' + data_split + '.doc.txt'
        raw_data = read_data_from(path, config.tokenizer, ace=not config.ere)
        data = remove_irrelevent_data(raw_data[:])
        gth_annotations = load_from_jsonl('data/ace_en/pt/'+ data_split +'.json')[0]
        meta_info = metadata.ace
        event_rep = get_event_rep(config.project_root + '/preprocess/ace/trigger_representation.json', 'event_name_seed')

    # Load trigger detection model
    config.pretrained_weights = 'bertlarge'
    config.EMBEDDING_DIM = 1024
    model_trigger = TriggerDetection(config)
    model_trigger.bert.resize_token_embeddings(len(config.tokenizer))
    model_trigger.load_state_dict(torch.load('saved_models/trigger_detection.pth'))
    model_trigger.cuda()

    # Load argument detection model
    from model.argument_detection.arg_detection import ModelRichContext
    model_arg = ModelRichContext(config)
    model_arg.bert.resize_token_embeddings(len(config.tokenizer))
    model_arg.load_state_dict(torch.load('saved_models/arg_detection.pth'))

    model_arg.cuda()
    # load argument query mapping and mask
    event_template_bert = {}
    event_types = sorted(list(event_rep.keys()))
    for e in event_types:
        temp = event_rep[e].split('-')
        event_template_bert[e] = token_to_berttokens(temp, config.tokenizer, template=True)
    arg_tokenizer_ids = arg_to_token_ids(meta_info.arg_set, config.tokenizer)
    arg_to_query_mask_dict = event_id_to_arg_query_and_mask(meta_info.trigger_arg_dic,
                                                            arg_tokenizer_ids,
                                                            meta_info.args_to_ids)

    # =============================================================================
    # Entity detection
    entity_to_ids = {'FAC': 0, 'GPE': 1, 'LOC': 2, 'ORG': 3, 'PER': 4, 'VEH': 5, 'WEA': 6, 'O': 7,
                     "[START]": 8, "[STOP]": 9, '[PAD]': 10}
    config.EMBEDDING_DIM = 768
    config.pretrained_weights='bert'
    model_entity = EntityDetection(entity_to_ids, config)
    model_entity.bert.resize_token_embeddings(len(config.tokenizer))
    model_entity.load_state_dict(torch.load('saved_models/entity_detection.pth'))
    model_entity.cuda()

    # initialize wrapper for generated dataset
    all_bert_tokens_arg, all_first_subword_idxs_arg, all_trigger_indicator, \
    all_bert_sentence_lengths, all_arg_mask, all_arg_type_ids, all_entity_mapping = [], [], [], [], [], [], []
    all_arg_tags = []
    SAVE_ARG_PT=args.save_arg_pt

    # Evaluation
    with torch.no_grad():
        all_preds = []
        for sent_id, this_data in enumerate(tqdm(data)):

            this_pred_str =  {'event_trigger': [], 'arg_list': []}
            # get entity spans
            this_data[0] = [x if config.tokenizer.tokenize(x) else ',' for x in this_data[0]]
            bert_tokens, to_collect = token_to_berttokens(['[CLS]']+this_data[0]+['[SEP]'], config.tokenizer)
            first_subword_idxs = torch.tensor([_ for _ in range(len(bert_tokens)) if to_collect[_]]).cuda().unsqueeze(0)
            bert_tokens = torch.tensor(config.tokenizer.convert_tokens_to_ids(bert_tokens)).cuda()

            feats, mask = model_entity(bert_tokens.unsqueeze(0), first_subword_idxs)
            pred = torch.argmax(feats, dim=-1)
            this_pred_entities = sorted(list(extract_spans(pred[0], entity_to_ids)))

            # get triggers:
            # CONTEXT: in the 1997 verdict , he was also convicted of sedition for his role in the 1979 military coup that brought him to power and a 1980 military crackdown that left hundreds of people dead in the southwestern city of kwangju .
            # input_sents: tokenized output of [CLS] EVENT_TYPE [SEP] SEED_TRIGGERS [SEP] CONTEXT [SET]
            # idxs_to_collect_event: the indices of EVENT_TYPE and SEED_TRIGGERS
            # idxs_to_collect_sent: the indices of CONTEXT in input_sents
            # pos_tags: POS tags of the CONTEXT, can be generated by the get_pos function
            trigger_inputs = pair_trigger_template([this_data], event_rep, config)
            trigger_inputs = dataset_prepare_trigger_zero_template(trigger_inputs, config, trigger_to_ids, metadata)
            (_, _, trigger_berttokens, idxs_to_collect_event, idxs_to_collect_sent, pos_tags, _) = trigger_inputs
            logits = model_trigger(None,
                                   trigger_berttokens.cuda().long(),
                                   idxs_to_collect_sent.cuda().long(),
                                   idxs_to_collect_event.cuda().long(), None, pos_tags.cuda().long())
            pred = ((logits[:, :, 1] - logits[:, :, 0] - config.trigger_threshold))
            pred = [pred[k] for k in range(config.event_count)]
            this_pred_trigger, this_pred_w_prob = pred_to_event_mention(pred, meta_info.ids_to_triggers, config)
            this_pred_str['event_trigger'] = this_pred_trigger

            # if no trigger detected
            this_pred_str['arg_list'] = []
            if not this_pred_trigger:
                all_preds.append(this_pred_str)
                continue

            # arg detection
            sent_len = len(this_data[0])
            for pred_trigger_i in this_pred_str['event_trigger']:
                arg_types = sorted(meta_info.trigger_arg_dic[pred_trigger_i[0]]) # prepare arg types for detected trigger
                input_text = ['[CLS]'] + this_data[0] + ['[SEP]'] + arg_types + ['O', '[SEP]'] # input context
                bert_tokens_arg, to_collect_arg = token_to_berttokens(input_text, config.tokenizer)
                first_sep_id = bert_tokens_arg.index('[SEP]')
                to_collect_arg = to_collect_arg[:first_sep_id]
                to_collect_arg = [i for i in range(len(to_collect_arg)) if to_collect_arg[i]>0] + [first_sep_id, first_sep_id]
                first_subword_idxs_arg = to_collect_arg
                bert_tokens_arg = config.tokenizer.convert_tokens_to_ids(bert_tokens_arg)
                trigger_indicator = [0 for _ in range(sent_len)]
                for _ in range(pred_trigger_i[1],pred_trigger_i[2]):
                    trigger_indicator[_] = 1
                bert_sentence_lengths = len(bert_tokens_arg)
                this_trigger_type_id = meta_info.triggers_to_ids[pred_trigger_i[0]]
                arg_mask = arg_to_query_mask_dict[this_trigger_type_id][1]
                arg_type_ids = arg_to_query_mask_dict[this_trigger_type_id][2]

                # create entity mapping matrix
                entity_indicator = (torch.ones((1, sent_len)).cuda().squeeze() * (-1.)).long()
                for j, x in enumerate(this_pred_entities):
                    entity_indicator[x[0]:x[1]] = j
                entity_mapping = from_entity_identifier_to_entity_matrix(entity_indicator.long())
                # gth argument annotations
                this_gth_tags_list = []
                this_gth_tags = {(x[2], x[3]): x[1] for x in gth_annotations[sent_id]['arg_list'] if
                                 x[0] == pred_trigger_i[0]}
                for j, x in enumerate(this_pred_entities):
                    if (x[0], x[1]) in this_gth_tags:
                        this_gth_tags_list.append(meta_info.args_to_ids[this_gth_tags[(x[0], x[1])].split('#@#')[0]])
                    else:
                        this_gth_tags_list.append(meta_info.args_to_ids['O'])

                if SAVE_ARG_PT:
                    # Save argument training data to .pt file with predicted triggers and entities
                    # Note that you may include gth triggers to improve the arg detection performance
                    all_bert_tokens_arg.append(bert_tokens_arg)
                    all_first_subword_idxs_arg.append(first_subword_idxs_arg)
                    all_trigger_indicator.append(trigger_indicator)
                    all_bert_sentence_lengths.append(bert_sentence_lengths)
                    all_arg_mask.append(arg_mask)
                    all_arg_type_ids.append(arg_type_ids)
                    all_entity_mapping.append(entity_mapping)

                    # gth arg tags:
                    this_gth_tags_list = []
                    this_gth_tags = {(x[2], x[3]): x[1] for x in gth_annotations[sent_id]['arg_list'] if x[0] == pred_trigger_i[0]}
                    for j, x in enumerate(this_pred_entities):
                        if (x[0], x[1]) in this_gth_tags:
                            this_gth_tags_list.append(meta_info.args_to_ids[this_gth_tags[(x[0], x[1])].split('#@#')[0]])
                        else:
                            this_gth_tags_list.append(meta_info.args_to_ids['O'])
                    all_arg_tags.append(torch.tensor(this_gth_tags_list))
                else:
                    # passing input to gpu
                    (bert_tokens_arg, first_subword_idxs_arg, trigger_indicator,
                     bert_sentence_lengths, arg_mask, arg_type_ids, entity_mapping) \
                        = (torch.tensor(x).unsqueeze(0).cuda()  for x in (
                                        bert_tokens_arg, first_subword_idxs_arg, trigger_indicator,
                                        bert_sentence_lengths, arg_mask, arg_type_ids, entity_mapping))
                    # arg detection forward
                    feats = model_arg(bert_tokens_arg, first_subword_idxs_arg, trigger_indicator,
                                      bert_sentence_lengths, arg_mask, arg_type_ids, entity_mapping)
                    # feats[:,:,-1] -= config.arg_threshold
                    pred = torch.argmax(feats, dim=-1)
                    pred += 1
                    pred = (pred * torch.round(torch.sum(entity_mapping, dim=1)).long() - 1).long()
                    # write arg predictionds
                    for j in range(len(this_pred_entities)):
                        if pred[0,j] < meta_info.args_to_ids['O']:
                            x = this_pred_entities[j]
                            this_pred_str['arg_list'].append([pred_trigger_i[0], meta_info.ids_to_args[pred[0, j].item()],  x[0], x[1]])

            this_pred_str['arg_list'] = list(map(tuple, this_pred_str['arg_list']))
            all_preds.append(this_pred_str)

        if SAVE_ARG_PT:
            # add paddings to the constructed training data
            all_bert_tokens_arg = pad_seq(all_bert_tokens_arg)
            all_first_subword_idxs_arg = pad_seq(all_first_subword_idxs_arg)
            all_trigger_indicator = pad_seq(all_trigger_indicator)
            all_bert_sentence_lengths = torch.tensor(all_bert_sentence_lengths).long().unsqueeze(1)
            all_arg_mask = torch.stack(list(map(torch.Tensor, all_arg_mask))).cuda()
            all_arg_type_ids = torch.Tensor(pad_sequences(all_arg_type_ids, maxlen=8, dtype="long",
                                                          truncating="post", padding="post",
                                                          value=meta_info.args_to_ids['[PAD]'])).long().cuda()
            all_entity_mapping = pad_seq(all_entity_mapping, dtype='float32')
            all_arg_tags = pad_seq(all_arg_tags, meta_info.args_to_ids['[PAD]'])
            # create dataset
            from torch.utils.data import DataLoader, TensorDataset
            all_data_to_save = TensorDataset(all_bert_tokens_arg, all_first_subword_idxs_arg, all_trigger_indicator,
                all_bert_sentence_lengths, all_arg_mask, all_arg_type_ids, all_entity_mapping, all_arg_tags)
            path = 'data/ere_en/processed_data/arg_' + data_split + '.pt'
            # save TensorDataset as pt file
            torch.save(all_data_to_save, path)
            print('Saving arg detection data to ', path)

        else:
            save_to_jsonl(all_preds, 'pred.json')

    # evaluation
    # all_preds = load_from_jsonl('pred.json') # load saved json file to compute F1 score
    pred_trigger = [[tuple(_) for _ in x['event_trigger']] for x in all_preds]
    gth_trigger = [[tuple(y) for y in x['event_trigger']] for x in gth_annotations]
    tp_trigger = sum([len(set(p).intersection(set(q))) for p, q in zip(pred_trigger, gth_trigger)])
    pos_trigger = len(sum(pred_trigger, []))
    gold_trigger = len(sum(gth_trigger, []))
    print(calculate_f1(gold_trigger, pos_trigger, tp_trigger))

    pred_args = [[tuple(_) for _ in x['arg_list']] for x in all_preds]
    gth_args = [[tuple(y) for y in x['arg_list']] for x in gth_annotations]
    tp_args = sum([len(set(p).intersection(set(q))) for p, q in zip(pred_args, gth_args)])
    pos_arg = sum([len(set(x)) for x in pred_args])
    gold_arg = sum([len(set(x)) for x in gth_args])
    print(calculate_f1(gold_arg, pos_arg, tp_args))



