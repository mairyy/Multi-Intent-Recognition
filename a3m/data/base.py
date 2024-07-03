import os
import logging
import csv
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BertTokenizer, AutoTokenizer

from .mm_pre import MMDataset
from .text_pre import TextDataset
from .video_pre import VideoDataset
from .audio_pre import AudioDataset
from .mm_pre import MMDataset
from .relation_pre import RelationDataset
from .__init__ import benchmarks

__all__ = ['DataManager']

class DataManager:
    
    def __init__(self, args, logger_name = 'Multimodal Intent Recognition'):
        
        self.logger = logging.getLogger(logger_name)

        self.benchmarks = benchmarks[args.dataset]

        self.data_path = os.path.join(args.data_path, args.dataset)

        if args.data_mode == 'multi-class':
            self.label_list = self.benchmarks["intent_labels"]
        elif args.data_mode == 'binary-class': 
            self.label_list = self.benchmarks['binary_intent_labels']
        else:
            raise ValueError('The input data mode is not supported.')
        self.logger.info('Lists of intent labels are: %s', str(self.label_list))

        args.num_labels = len(self.label_list)        
        args.text_feat_dim, args.video_feat_dim, args.audio_feat_dim = \
            self.benchmarks['feat_dims']['text'], self.benchmarks['feat_dims']['video'], self.benchmarks['feat_dims']['audio']
        args.text_seq_len, args.video_seq_len, args.audio_seq_len = \
            self.benchmarks['max_seq_lengths']['text'], self.benchmarks['max_seq_lengths']['video'], self.benchmarks['max_seq_lengths']['audio']

        if args.method == 'shark' or args.method == 'a3m':
            args.relation_seq_len = self.benchmarks['max_seq_lengths']['relation']
            args.relation_feat_dim = args.text_feat_dim
            
        if args.text_backbone.startswith('bart'):
            self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
            additional_special_tokens = ['[CLS]', '[SEP]', '[BIMG]', '[EIMG]', '[IFEAT]']
            self.tokenizer.add_tokens(additional_special_tokens)  
            # unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
            # self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + additional_special_tokens
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            additional_special_tokens = ['[BIMG]', '[EIMG]', '[IFEAT]']
            self.tokenizer.add_tokens(additional_special_tokens) 
        
        self.train_data_index, self.train_label_ids = self._get_indexes_annotations(os.path.join(self.data_path, 'train.tsv'), args.data_mode)
        self.dev_data_index, self.dev_label_ids = self._get_indexes_annotations(os.path.join(self.data_path, 'dev.tsv'), args.data_mode)
        self.test_data_index, self.test_label_ids = self._get_indexes_annotations(os.path.join(self.data_path, 'test.tsv'), args.data_mode)

        self.unimodal_feats = self._get_unimodal_feats(args, self._get_attrs())
        self.mm_data = self._get_multimodal_data(args)
        self.mm_dataloader = self._get_dataloader(args, self.mm_data)

    def _get_indexes_annotations(self, read_file_path, data_mode):

        label_map = {}
        for i, label in enumerate(self.label_list):
            label_map[label] = i

        with open(read_file_path, 'r') as f:

            data = csv.reader(f, delimiter="\t")
            indexes = []
            label_ids = []

            for i, line in enumerate(data):
                if i == 0:
                    continue
                
                index = '_'.join([line[0], line[1], line[2]])
                indexes.append(index)
                
                if data_mode == 'multi-class':
                    label_id = label_map[line[4]]
                else:
                    label_id = label_map[self.benchmarks['binary_maps'][line[4]]]
                
                label_ids.append(label_id)

        return indexes, label_ids
    
    def _get_unimodal_feats(self, args, attrs):
        
        text_feats = TextDataset(args, attrs).feats
        video_feats = VideoDataset(args, attrs).feats
        audio_feats = AudioDataset(args, attrs).feats
        comet_relation_feats = RelationDataset(args, attrs, 'comet').feats
        sbert_relation_feats = RelationDataset(args, attrs, 'sbert').feats

        return {
            'text': text_feats,
            'video': video_feats,
            'audio': audio_feats,
            'relation': {
                'comet': comet_relation_feats,
                'sbert': sbert_relation_feats
            }
        }
    
    def _get_multimodal_data(self, args):

        text_data = self.unimodal_feats['text']
        video_data = self.unimodal_feats['video']
        audio_data = self.unimodal_feats['audio']
        comet_data = self.unimodal_feats['relation']['comet']
        sbert_data = self.unimodal_feats['relation']['sbert']
        
        mm_train_data = MMDataset(self.train_label_ids, text_data['train'], video_data['train'],\
                                audio_data['train'], comet_data['train'], sbert_data['train'])
        mm_dev_data = MMDataset(self.dev_label_ids, text_data['dev'], video_data['dev'], \
                                audio_data['dev'], comet_data['dev'], sbert_data['dev'])
        mm_test_data = MMDataset(self.test_label_ids, text_data['test'], video_data['test'], \
                                 audio_data['test'], comet_data['test'], sbert_data['test'])

        return {
            'train': mm_train_data,
            'dev': mm_dev_data,
            'test': mm_test_data
        }

    def _get_dataloader(self, args, data):
        
        self.logger.info('Generate Dataloader Begin...')

        train_dataloader = DataLoader(data['train'], shuffle=True, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)
        dev_dataloader = DataLoader(data['dev'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)
        test_dataloader = DataLoader(data['test'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)

        self.logger.info('Generate Dataloader Finished...')

        return {
            'train': train_dataloader,
            'dev': dev_dataloader,
            'test': test_dataloader
        }
        
    def _get_attrs(self):

        attrs = {}
        for name, value in vars(self).items():
            attrs[name] = value

        return attrs


