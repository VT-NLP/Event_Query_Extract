# coding=utf-8
import torch
import os
from transformers import BertTokenizer, AutoTokenizer, RobertaTokenizer, DistilBertTokenizer


class Config(object):
    def __init__(self):
        self.project_root = ''
        self.data_path = self.project_root + '/data/ace_en/pt/'
        self.torch_seed = 39

        # file path
        self.tr_dataset = self.data_path + 'train.pt'
        self.te_dataset = self.data_path + 'test.pt'
        self.dev_dataset = self.data_path + 'dev.pt'
        self.save_model_path = self.project_root + 'saved_models/'
        self.error_visualization_path = self.project_root + 'error_visualizations/'
        self.pretrain_model_id = ''
        self.pretrained_model_path = ''
        self.train_file = ''
        self.dev_file = ''
        self.test_file = ''
        self.joint_train_pt = ''
        self.save_to_json_stats = ''
        self.event_rep = None
        self.log_file = ''
        self.te_json = ''
        self.dev_json = self.data_path + 'dev.json'
        self.te_json = self.data_path + 'test.json'
        self.tr_json = self.data_path + 'train.json'
        self.output_json = self.project_root + 'outputs/output.json'
        self.test_pred_trigger_json = ''
        self.dev_pred_trigger_json = ''

        # device info
        self.use_gpu = True
        self.device = None

        # model parameters
        self.pretrained_weights = 'bert'
        self.tokenizer = None
        self.same_bert = False
        self.freeze_bert = False
        self.EMBEDDING_DIM = 768
        self.ENTITY_EMB_DIM = 100
        self.extra_bert = -3
        self.use_extra_bert = False
        self.n_hid = 200
        self.dropout = 0.5
        self.ere = False
        self.do_train = True
        self.event_f1 = False
        self.load_pretrain = False
        self.arg_joint = False
        self.last_k_hidden = 3
        self.trigger_threshold = 0.65

        # optimizer parameters
        self.trigger_training_weight = 1
        self.non_weight = 1
        self.BATCH_SIZE = 16
        self.EPOCH = 5
        self.lr = 5e-5
        self.weight_decay = 0.001
        self.warmup_proportion = 0.1
        self.gradient_accumulation_steps = 1
        self.eval_per_epoch = 10
        self.sampling = 0.5
        self.loss_type = 'sum'

        # data details
        self.arg_count = 22
        self.event_count = 33
        self.entity_count = 8
        self.pos_count = 17
        self.fact_container = None
        self.arg_roles = 22
        self.entity_roles = 6
        self.PAD_TAG = '[PAD]'
        self.gold_arg_count = 0
        self.eps = 1e-6
        self.gold_arg_count = 689

        self.extra_setup()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def extra_setup(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_gpu else "cpu")
        self.set_tokenizer()
        self.resize_tokenizer()
        self.pretrained_model_path = self.save_model_path + self.pretrain_model_id
        self.tr_dataset = self.data_path + 'train.pt'
        self.te_dataset = self.data_path + 'test.pt'
        self.dev_dataset = self.data_path + 'dev.pt'
        self.dev_pred_trigger_json = self.project_root + '/outputs/' + self.dev_pred_trigger_json
        self.test_pred_trigger_json = self.project_root + '/outputs/' + self.test_pred_trigger_json

    def resize_tokenizer(self):
        special_tokens_dict = {'additional_special_tokens': ['<entity>', '</entity>', '<event>', '</event>', '[EVENT]']}
        self.tokenizer.add_special_tokens(special_tokens_dict)

    def set_tokenizer(self):
        if self.pretrained_weights == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        elif self.pretrained_weights == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif self.pretrained_weights == 'bertlarge':
            self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        elif self.pretrained_weights == 'spanbert':
            self.tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-large-cased")
        elif self.pretrained_weights == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        else:
            self.tokenizer = None

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])

