from torch.utils.data import DataLoader
import json
import torch


def load_data_from_pt(dataset, batch_size, shuffle=False):
    """
    Load data saved in .pt
    :param dataset:
    :param batch_size:
    :param shuffle:
    :return:
    """
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)


def pack_data_to_trigger_model_joint(batch):
    """
    Prepare data for trigger detection model
    :param batch:
    :return:
    """
    # unpack and truncate data
    (_, bert_sentence_lengths,
     bert_tokens, idxs_to_collect_event, idxs_to_collect_sent, pos_tags, event_tags) = batch
    embedding_length = int(torch.max(bert_sentence_lengths).item())
    bert_sentence_in = bert_tokens[:, :embedding_length]
    idxs_to_collect_sent = idxs_to_collect_sent[:, :embedding_length]
    sent_lengths = torch.sum(idxs_to_collect_sent, dim=1).int()
    max_sent_len = int(torch.max(sent_lengths).item())
    trigger_tags = event_tags[:, :max_sent_len]
    pos_tags = pos_tags[:, :max_sent_len]

    # send data to gpu
    tmp = [bert_sentence_in, trigger_tags, idxs_to_collect_event, idxs_to_collect_sent, sent_lengths, \
           bert_sentence_lengths, pos_tags]
    return [x.long() for x in tmp] + [embedding_length]


def calculate_f1(gold, pred, tp, eps=1e-6):
    recall = tp / (gold + eps)
    precision = tp / (pred + eps)
    f1 = 2 * (recall * precision) / (recall + precision + eps)
    return f1, precision, recall


def pack_data_to_arg_model(batch):
    # unpack batch
    (dataset_id, sent_idx_all, sent_lengths, bert_sentence_lengths,
     bert_tokens, first_subword_idxs, pred_indicator,
     arg_tags, arg_mask, arg_type_ids,
     pos_tags, entity_identifier, entity_tags, entity_mask, entity_mapping,
     trigger_feats) = batch

    # simple processing
    trigger_indicator = pred_indicator
    max_bert_len = torch.max(bert_sentence_lengths)
    max_sent_len = max([sum(first_subword_idxs[i] > 0) for i in range(len(first_subword_idxs))]) - 2

    # to cuda
    (dataset_id, sent_idx_all, sent_lengths, bert_sentence_lengths,
     bert_tokens, first_subword_idxs, pred_indicator, arg_tags,
     arg_type_ids,
     pos_tags, entity_identifier, entity_tags, entity_mask
     ) = tuple(map(torch.Tensor.long, (dataset_id, sent_idx_all, sent_lengths, bert_sentence_lengths,
                                       bert_tokens, first_subword_idxs, pred_indicator, arg_tags,
                                       arg_type_ids,
                                       pos_tags, entity_identifier, entity_tags, entity_mask
                                       )))
    # slice1, one dimension data
    dataset_id = dataset_id[:max_bert_len]
    sent_idx_all = sent_idx_all[:max_bert_len]
    bert_sentence_lengths = bert_sentence_lengths[:max_bert_len]
    # slice2, 2d, bs x bert_len
    bert_tokens = bert_tokens[:, :max_bert_len]
    # slice3, 2d, bs x sent_len
    first_subword_idxs = first_subword_idxs[:, :max_sent_len + 2]
    trigger_indicator = trigger_indicator[:, :max_sent_len]
    arg_tags = arg_tags[:, :max_sent_len]
    arg_type_ids = arg_type_ids[:, :max_sent_len]
    pos_tags = pos_tags[:, :max_sent_len]
    entity_identifier = entity_identifier[:, :max_sent_len]
    entity_tags = entity_tags[:, :max_sent_len]
    entity_mask = entity_mask[:, :max_sent_len]
    entity_mapping = entity_mapping[:, :max_sent_len]
    first_subword_idxs -= 2
    return dataset_id, sent_idx_all, bert_sentence_lengths, bert_tokens, first_subword_idxs, \
           trigger_indicator, arg_tags, arg_mask, arg_type_ids, \
           pos_tags, entity_identifier, entity_tags, entity_mask, entity_mapping, trigger_feats


def save_to_json(json_output, path):
    jsonString = json.dumps(json_output)
    jsonFile = open(path, "w")
    jsonFile.write(jsonString)
    jsonFile.close()
    return


def save_to_jsonl(json_output, path):
    with open(path, 'w') as outfile:
        for entry in json_output:
            json.dump(entry, outfile)
            outfile.write('\n')
    outfile.close()
    return


def load_from_jsonl(json_path):
    data = []
    with open(json_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def pred_to_event_mention(pred, ids_to_triggers, config):
    ret = []
    for i in range(config.event_count):

        if not torch.any(pred[i]>0.5):
            continue

        temp = torch.cat([torch.tensor([0]).cuda(), pred[i], torch.tensor([0]).cuda()])
        is_event, begin, end = 0, None, None
        for j in range(len(temp)):
            if temp[j] and not is_event:
                begin = j-1
                is_event = 1
            if not temp[j] and is_event:
                end = j-1
                is_event = 0
                ret.append(tuple([ids_to_triggers[i], begin, end]))
    return ret