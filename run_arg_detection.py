import time
import torch
import os
import sys
import torch.nn as nn
from model.argument_detection.arg_detection import ModelRichContext
from utils.config import Config
from utils.to_html import Write2HtmlArg
import fire
from utils.utils_model import *
import logging
from utils.optimization import BertAdam, warmup_linear
from datetime import datetime
from utils.fact_container import FactContainer
from torch.utils.data import DataLoader
import copy


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    filename=os.getenv('LOGFILE'))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def load_data_from_pt(dataset, batch_size, shuffle=False):
    """
    Load data saved in .pt
    :param dataset:
    :param batch_size:
    :param shuffle:
    :return:
    """
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)


def read_sent_from(path):
    """
    Read sentences from the given path
    :param path:
    :return:
    """
    f = open(path, 'r')
    data = f.read().splitlines()
    for i in range(len(data)):
        data[i] = data[i].split(' ')
    return data


def main(**kwargs):
    # configuration
    config = Config()
    config.update(**kwargs)
    logging.info(config)
    torch.backends.cudnn.enabled = False
    torch.manual_seed(39)
    config.fact_container = FactContainer()

    # load data
    tr_dataset = torch.load(config.train_file_pt)
    dev_dataset = torch.load(config.dev_file_pt)
    dev_json = json.load(open(config.dev_json))

    # =================================== Arg Model ==========================================
    model_arg = ModelRichContext(config)
    model_arg.bert.resize_token_embeddings(len(config.tokenizer))
    model_arg.cuda()
    if config.load_pretrain:
        model_arg.load_state_dict(torch.load(config.pretrained_model_path))

    # optimizer
    param_optimizer1 = list(model_arg.bert.named_parameters())
    param_optimizer2 = list(model_arg.linear.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer2 if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay * 10, 'lr': config.lr * 10},
        {'params': [p for n, p in param_optimizer2 if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    N_train = len(tr_dataset)
    num_train_steps = int(N_train / config.BATCH_SIZE / config.gradient_accumulation_steps * config.EPOCH)
    t_total = num_train_steps
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.lr,
                         warmup=config.warmup_proportion,
                         t_total=t_total,
                         weight_decay=0)

    # loss
    weights = torch.ones(config.arg_roles + 1).cuda()
    weights[-1] = config.non_weight
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=config.arg_roles + 1)

    global_step = [0]
    f1, pre_f1 = 0, 0
    best_model = model_arg
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=config.BATCH_SIZE)

    # training
    for epoch in range(config.EPOCH):
        logging.info('==============')
        tr_loader = DataLoader(tr_dataset, shuffle=True, batch_size=int(config.BATCH_SIZE))
        model_new_best, f1 = train_arg(config, model_arg, epoch, tr_loader, criterion, optimizer, t_total,
                                       global_step, pre_f1, dev_json, dev_loader)
        # save best model if achieve better F1 score on dev set
        if f1 > pre_f1 and model_new_best:
            best_model = copy.deepcopy(model_new_best)
            pre_f1 = f1
    # save best model
    date_time = datetime.now().strftime("%m%d%Y%H:%M:%S")
    logging.info('Save best model to {}'.format(config.save_model_path + date_time))
    torch.save(best_model.state_dict(), config.save_model_path + date_time)

    te_dataset = torch.load(config.test_file_pt)
    te_json = json.load(open(config.te_json))
    te_loader = DataLoader(te_dataset, shuffle=False, batch_size=config.BATCH_SIZE)
    f1, precision, recall, json_output = eval_arg(best_model, te_json, te_loader, config)
    logging.info('Test f1_bio: {} |  p:{}  | r:{}'.format(f1, precision, recall))
    return 0


def train_arg(config, model, epoch, tr_loader, criterion, optimizer, t_total, global_step, pre_f1,
              eval_json=None, eval_loader=None):
    model.train()
    model.zero_grad()
    f1_new_best, model_new_best = pre_f1, None
    eval_step = int(len(tr_loader) / config.eval_per_epoch)
    for i, batch in enumerate(tr_loader):
        # Extract data
        dataset_id, sent_idx, bert_sentence_lengths, bert_tokens, first_subword_idxs, \
        trigger_indicator, arg_tags, arg_mask, arg_type_ids, \
        pos_tags, entity_identifier, entity_tags, entity_mask, entity_mapping, trigger_feats \
            = pack_data_to_arg_model(batch)

        # forward
        feats = model(bert_tokens, first_subword_idxs, trigger_indicator,
                      bert_sentence_lengths, arg_mask, arg_type_ids, entity_mapping)

        # Loss
        feats = feats[:, :arg_tags.shape[1]]
        logits_padded = feats.flatten(start_dim=0, end_dim=-2)
        targets = arg_tags.flatten().long()
        loss = criterion(logits_padded, targets)
        loss.backward()

        # learning rate warm up
        if (i + 1) % config.gradient_accumulation_steps == 0:
            rate = warmup_linear(global_step[0] / t_total, config.warmup_proportion)
            for param_group in optimizer.param_groups[:-2]:
                param_group['lr'] = config.lr * rate
            for param_group in optimizer.param_groups[-2:]:
                param_group['lr'] = config.lr * rate * 20
            optimizer.step()
            optimizer.zero_grad()
            global_step[0] += 1

        if (i + 1) % eval_step == 0 :
            f1, precision, recall, output = eval_arg(model, eval_json, eval_loader, config)
            if f1 > pre_f1:
                model_new_best = copy.deepcopy(model)
                pre_f1 = f1
                f1_new_best = f1
                logging.info('New best result found for Dev.')
                logging.info('epoch: {} | f1_bio: {} |  p:{}  | r:{}'.format(epoch, f1, precision, recall))

    return model_new_best, f1_new_best

def eval_arg(model, gold_event, dev_loader, config, mode='dev'):
    model.eval()
    ids_to_arg = config.fact_container.ids_to_arg28
    ids_to_trigger = config.fact_container.ix_to_trigger
    to_html = Write2HtmlArg(ids_to_arg, ids_to_trigger, config.arg_roles)
    json_output = []
    gold_all = []
    pred_all = []
    feats_all = []
    entity_identifier_all = []
    trigger_indicator_all = []
    with torch.no_grad():
        for i, batch in enumerate(dev_loader):
            # Extract data
            dataset_id, sent_idx, bert_sentence_lengths, bert_tokens, first_subword_idxs, \
            trigger_indicator, arg_tags, arg_mask, arg_type_ids, \
            pos_tags, entity_identifier, entity_tags, entity_mask, entity_mapping, trigger_feats \
                = pack_data_to_arg_model(batch)
            first_subword_idxs -= 2

            # forward
            feats = model(bert_tokens, first_subword_idxs, trigger_indicator,
                          bert_sentence_lengths, arg_mask, arg_type_ids, entity_identifier, entity_tags)

            # gather predictions and true labels
            feats[entity_mask == 0] = -1e6
            pred = torch.argmax(feats, dim=-1)
            pred_all.extend(pred)
            gold_all.extend(arg_tags)
            feats_all.extend(feats)
            entity_identifier_all.extend(entity_identifier)
            trigger_indicator_all.extend(trigger_indicator)

        # log and generate error visualization in .html
        now = datetime.now()
        date_time = now.strftime("%m%d%Y%H:%M:%S")
        to_html.arg_to_html(gold_event, gold_all, pred_all, feats_all, entity_identifier_all)
        to_html.write_to(config.error_visualization_path + '/arg_error.html', trigger_indicator_all)
        tp1, fp1 = to_html.write_to_one_ie_f1()
        precision = tp1 / (tp1 + fp1 + config.eps)
        recall = tp1 / config.gold_arg_count
        f1 = 2 * (precision * recall) / (precision + recall + config.eps)

    return f1, precision, recall, json_output


if __name__ == '__main__':
    fire.Fire()
