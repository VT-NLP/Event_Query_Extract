import time
import torch
import torch.nn as nn
from model.trigger_detection.trigger_detection import TriggerDetection
from utils.config import Config
from utils.utils_model import *
import fire
import logging
from utils.optimization import BertAdam, warmup_linear
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import json
import copy
from tqdm import tqdm
import sys
import os
from utils.metadata import Metadata

eps = 1e-6


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    filename=os.getenv('LOGFILE'))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def main(**kwargs):
    # configuration
    config = Config()
    config.update(**kwargs)
    config.extra_setup()
    logging.info(config)
    torch.backends.cudnn.enabled = False
    torch.manual_seed(config.torch_seed)
    metadata = Metadata()

    # load data
    te_dataset = torch.load(config.te_dataset)
    dev_dataset = torch.load(config.dev_dataset)
    tr_dataset = torch.load(config.tr_dataset)
    te_json = json.load(open(config.te_json))
    dev_json = json.load(open(config.dev_json))
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=config.event_count)
    te_loader = DataLoader(te_dataset, shuffle=False, batch_size=config.event_count)

    # model setup
    model_trigger = TriggerDetection(config)
    model_trigger.bert.resize_token_embeddings(len(config.tokenizer))
    if config.use_gpu:
        model_trigger.cuda()
    if config.load_pretrain:
        model_trigger.load_state_dict(torch.load(config.pretrained_model_path))

    # =======================================================================
    # optimizer
    param_optimizer1 = list(model_trigger.bert.named_parameters())
    param_optimizer1 = [n for n in param_optimizer1 if 'pooler' not in n[0]]
    param_optimizer2 = list(model_trigger.linear.named_parameters())
    param_optimizer2.append(('W', model_trigger.W))
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer2 if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay, 'lr': config.lr * 10},
        {'params': [p for n, p in param_optimizer2 if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    N_train = len(tr_dataset)
    num_train_steps = int(
        N_train * config.sampling / config.BATCH_SIZE / config.gradient_accumulation_steps * config.EPOCH)
    t_total = num_train_steps
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.lr,
                         warmup=config.warmup_proportion,
                         t_total=t_total)

    # loss
    weights = torch.ones(2).cuda()
    weights[0] = config.non_weight
    weights[1] = config.trigger_training_weight
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1, reduction='mean')

    f1, pre_f1 = 0, 0
    global_step = [0]

    best_model = model_trigger
    for epoch in range(config.EPOCH):
        logging.info('==============')
        logging.info('Training at ' + str(epoch) + ' epoch')

        # sample instances to reduce training time
        pos_ids = torch.tensor([torch.any(x[-1] < config.event_count) for x in tr_dataset])
        random_neg_ids = torch.rand(pos_ids.shape) < config.sampling
        _tr_dataset = TensorDataset(*tr_dataset[pos_ids + random_neg_ids])
        tr_loader = DataLoader(_tr_dataset, shuffle=True, batch_size=int(config.BATCH_SIZE))

        # training
        model_new_best, f1 = train_trigger(config, model_trigger, epoch, pre_f1,
                                           tr_loader, criterion, optimizer, t_total, global_step, metadata ,dev_loader, dev_json)

        # save best model if achieve better F1 score on dev set
        if f1 > pre_f1 and model_new_best:
            best_model = copy.deepcopy(model_new_best)
            pre_f1 = f1
    f1, precision, recall, json_output = eval_trigger(best_model, te_loader, config, metadata, te_json)
    date_time = datetime.now().strftime("%m%d%Y%H:%M:%S")
    logging.info('Save best model to {}'.format(config.save_model_path + date_time))
    torch.save(best_model.state_dict(), config.save_model_path + date_time)
    logging.info('Test')
    logging.info('f1_bio: {} |  p:{}  | r:{}'.format(f1, precision, recall))
    save_to_jsonl(json_output, 'outputs/'+date_time+'.json')
    return 0


def train_trigger(config, model, epoch, pre_f1, tr_loader, criterion, optimizer, t_total, global_step, metadata,
                  eval_loader=None, gold_event=None):
    logging.info("Begin trigger training...")
    logging.info("Bert model: {}\nBatch size: {}".format(str(config.pretrained_weights), config.BATCH_SIZE))
    logging.info("Epoch {}".format(epoch))
    logging.info("time: {}".format(time.asctime()))

    # model.base_type_dataset()
    model.zero_grad()
    f1_new_best, model_new_best = pre_f1, None

    num_batchss = len(tr_loader)
    eval_step = int(num_batchss / config.eval_per_epoch)
    for i, batch in enumerate(tqdm(tr_loader)):
        # Extract data
        bert_sentence_in, triggers, idxs_to_collect_event, idxs_to_collect_sent, sent_lengths, \
        bert_sentence_lengths, pos_tags, embedding_length, \
            = pack_data_to_trigger_model_joint(batch)

        # forward
        feats = model(torch.zeros_like(sent_lengths), bert_sentence_in,
                      idxs_to_collect_sent, idxs_to_collect_event, bert_sentence_lengths, pos_tags)

        # Loss
        triggers = triggers.flatten()
        feats = torch.flatten(feats, end_dim=-2)
        targets = triggers[triggers != config.event_count + 1]
        targets = (targets < config.event_count) * 1
        feats = feats[triggers != config.event_count + 1]

        loss = criterion(feats, targets)
        loss.backward()

        # modify learning rate with special warm up BERT uses
        if (i + 1) % config.gradient_accumulation_steps == 0:
            rate = warmup_linear(global_step[0] / t_total, config.warmup_proportion)
            for param_group in optimizer.param_groups[:-2]:
                param_group['lr'] = config.lr * rate
            for param_group in optimizer.param_groups[-2:]:
                param_group['lr'] = config.lr * 20 * rate
            optimizer.step()
            optimizer.zero_grad()
            global_step[0] += 1

        if (i + 1) % eval_step == 0 and eval_loader and epoch > 0:
            f1, precision, recall, output = eval_trigger(model, eval_loader, config, metadata, gold_event)
            if f1 > pre_f1:
                model_new_best = copy.deepcopy(model)
                pre_f1 = f1
                f1_new_best = f1
                logging.info('New best result found for Dev.')
                logging.info('epoch: {} | f1_bio: {} |  p:{}  | r:{}'.format(epoch, f1, precision, recall))

    return model_new_best, f1_new_best


def eval_trigger(model, eval_loader, config, metadata, gold_event):
    model.eval()
    output = []
    # evaluate
    tp, pos, gold = 0, 0, 0
    if config.ere:
        ids_to_triggers = metadata.ere.ids_to_triggers
    else:
        ids_to_triggers = metadata.ace.ids_to_triggers
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            # Extract data
            bert_sentence_in, triggers, idxs_to_collect_event, idxs_to_collect_sent, sent_lengths, \
            bert_sentence_lengths, pos_tags, embedding_length, \
                = pack_data_to_trigger_model_joint(batch)

            # forward
            logits = model(torch.zeros_like(sent_lengths), bert_sentence_in,
                          idxs_to_collect_sent, idxs_to_collect_event, bert_sentence_lengths, pos_tags)

            # get predictions from logits
            pred = ((logits[:,:,1] - logits[:,:,0] - config.trigger_threshold)>0)*1
            pred = [pred[k,:sent_lengths[k]] for k in range(config.event_count)]

            this_pred = (set(pred_to_event_mention(pred, ids_to_triggers, config)))
            this_gold = set(tuple(x) for x in gold_event[i]['event_trigger'])
            tp += len(this_gold.intersection(this_pred))
            pos += len(this_pred)
            gold += len(this_gold)
            output.append({"pred":list(this_pred)})
    f1, precision, recall = calculate_f1(gold, pos, tp)
    # print(gold, pos, tp)
    model.train()
    return f1, precision, recall, output


if __name__ == '__main__':
    fire.Fire()
