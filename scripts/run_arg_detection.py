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
from torch.utils.data import DataLoader
import copy


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    filename=os.getenv('LOGFILE'))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def main(**kwargs):
    # configuration
    config = Config()
    config.update(**kwargs)
    logging.info(config)
    torch.backends.cudnn.enabled = False
    torch.manual_seed(39)


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

    # load training dataset
    tr_dataset = torch.load(config.tr_dataset)
    dev_dataset = torch.load(config.dev_dataset)

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
                                       global_step, pre_f1, dev_loader)

        # save best model if achieve better F1 score on dev set
        if f1 > pre_f1 and model_new_best:
            best_model = copy.deepcopy(model_new_best)
            pre_f1 = f1
    # save best model
    date_time = datetime.now().strftime("%m%d%Y%H:%M:%S")
    logging.info('Save best model to {}'.format(config.save_model_path + date_time))
    torch.save(model_arg.state_dict(), config.save_model_path + date_time)

    te_dataset = torch.load(config.te_dataset)
    te_loader = DataLoader(te_dataset, shuffle=False, batch_size=config.BATCH_SIZE)
    f1, precision, recall, json_output = eval_arg(best_model, te_loader, config)
    logging.info('Test f1_bio: {} |  p:{}  | r:{}'.format(f1, precision, recall))
    return 0


def train_arg(config, model, epoch, tr_loader, criterion, optimizer, t_total, global_step, pre_f1,
              eval_loader=None):
    model.train()
    model.zero_grad()
    f1_new_best, model_new_best = pre_f1, None
    eval_step = int(len(tr_loader) / config.eval_per_epoch)
    for i, batch in enumerate(tr_loader):
        # Extract data
        bert_tokens, first_subword_idxs, trigger_indicator,\
        bert_sentence_lengths, arg_mask, arg_type_ids, entity_mapping, arg_tags = batch
        # forward
        feats = model(bert_tokens.long(), first_subword_idxs.long(), trigger_indicator.long(),
                      bert_sentence_lengths.long(), arg_mask, arg_type_ids.long(), entity_mapping)

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
            # print(loss.item())
            f1, precision, recall, output = eval_arg(model, eval_loader, config)
            if f1 > pre_f1:
                model_new_best = copy.deepcopy(model)
                pre_f1 = f1
                f1_new_best = f1
            logging.info('New best result found for Dev.')
            logging.info('epoch: {} | f1_bio: {} |  p:{}  | r:{}'.format(epoch, f1, precision, recall))

    return model_new_best, f1_new_best

def eval_arg(model, dev_loader, config, mode='dev'):
    model.eval()
    json_output = []
    tp, pos, gold = 0, 0 ,0
    with torch.no_grad():
        for i, batch in enumerate(dev_loader):
            # Extract data
            bert_tokens, first_subword_idxs, trigger_indicator, \
            bert_sentence_lengths, arg_mask, arg_type_ids, entity_mapping, arg_tags = batch
            # forward
            feats = model(bert_tokens.long(), first_subword_idxs.long(), trigger_indicator.long(),
                          bert_sentence_lengths.long(), arg_mask, arg_type_ids.long(), entity_mapping)

            # gather predictions and true labels
            pred = torch.argmax(feats, dim=-1)
            pred += 1
            pred = (pred * torch.round(torch.sum(entity_mapping, dim=1)).long() -1).long()
            arg_tags = torch.round(arg_tags).long()
            tp+= torch.sum(torch.logical_and(pred[:, :arg_tags.shape[1]] ==arg_tags, arg_tags!=config.arg_roles))
            pos+= torch.sum(torch.logical_and(pred <config.arg_roles, pred > -0.5))
            gold+= torch.sum(arg_tags<config.arg_roles)
        # pseudo F1 score calculation, please refer to eval.py for exact F1 score calculation
        now = datetime.now()
        date_time = now.strftime("%m%d%Y%H:%M:%S")
        precision = tp / (pos + config.eps)
        recall = tp / config.gold_arg_count
        f1 = 2 * (precision * recall) / (precision + recall + config.eps)

    return f1, precision, recall, json_output


if __name__ == '__main__':
    fire.Fire()
