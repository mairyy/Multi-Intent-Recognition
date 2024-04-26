import os
import csv
import sys
import logging
import pandas as pd
from transformers import BertTokenizer

__all__ = ['TextDataset']

class TextDataset:
    
    def __init__(self, args, base_attrs):
        
        self.logger = logging.getLogger(args.logger_name)
        self.base_attrs = base_attrs
        
        if args.text_backbone.startswith('bert'):
            self.feats = self._get_feats(args, base_attrs)
        else:
            raise Exception('Error: inputs are not supported text backbones.')

    def _get_feats(self, args, base_attrs):

        self.logger.info('Generate Text Features Begin...')

        processor = DatasetProcessor()

        train_examples= processor.get_examples(base_attrs['data_path'], 'train', args.relation, args.relation_type)
        train_feats = self._get_bert_feats(args, train_examples, base_attrs, 'train')

        dev_examples = processor.get_examples(base_attrs['data_path'], 'dev', args.relation, args.relation_type)
        dev_feats = self._get_bert_feats(args, dev_examples, base_attrs, 'dev')

        test_examples = processor.get_examples(base_attrs['data_path'], 'test', args.relation, args.relation_type)
        test_feats = self._get_bert_feats(args, test_examples, base_attrs, 'test')
        
        self.logger.info('Generate Text Features Finished...')

        return {
            'train': train_feats,
            'dev': dev_feats,
            'test': test_feats,
        }

    def _get_bert_feats(self, args, examples, base_attrs, type_mode='test', relation=None):

        max_seq_length = base_attrs["benchmarks"]['max_seq_lengths']['text']
        max_relation_length = base_attrs["benchmarks"]['max_seq_lengths']['relation']
        max_seq_length += max_relation_length

        if args.text_backbone.startswith('bert'):
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)   

        features = convert_examples_to_features(examples, max_seq_length, tokenizer)     
        features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in features]

        # relation_features = convert_examples_to_features(relation, max_relation_length, tokenizer)
        # relation_features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in relation_features]

        return features_list

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        inputs = pd.read_csv(input_file, sep='\t')
        return inputs
    #     with open(input_file, "r") as f:
    #         reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
    #         lines = []
    #         for line in reader:
    #             if sys.version_info[0] == 2:
    #                 line = list(unicode(cell, 'utf-8') for cell in line)
    #             lines.append(line)
    #         return lines
    
    @classmethod
    def _read_csv(cls, input_file):
        """Read csv file. Return pandas dataframe."""
        inputs = pd.read_csv(input_file, sep=',')
        return inputs

class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode, relation=False, relation_type=None):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir, relation, relation_type)
        elif mode == 'dev':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir, relation, relation_type)
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", data_dir, relation, relation_type)

    def _create_examples(self, lines, set_type, data_dir=None, relation=False, relation_type=None):
        """Creates examples for the training and dev sets."""
        examples = []
        # relations = None
        texts = lines['text']
        # for (i, line) in enumerate(lines):
        #     if i == 0:
        #         continue

        #     guid = "%s-%s" % (set_type, i)
        #     text_a = line[3]

        #     examples.append(
        #         InputExample(guid=guid, text_a=text_a, text_b=None))
        if relation:
            # relations = []
            if set_type == "train":
                inputs = self._read_csv(os.path.join(data_dir, "relations", "atomic_train.csv"))
            elif set_type == "dev":
                inputs = self._read_csv(os.path.join(data_dir, "relations", "atomic_dev.csv"))
            else:
                inputs = self._read_csv(os.path.join(data_dir, "relations", "atomic_test.csv"))
            
            for i, line in enumerate(inputs[relation_type]):
                guid = "%s-%s" % (set_type, i+1)
                if relation_type == 'xAttr':
                    line = "the part of speaker is " + line[2:len(line)-2]
                elif relation_type == 'xReact':
                    line = "the reaction of speaker is " + line[2:len(line)-2]
                elif relation_type == 'xWant':
                    # line = "the intention of speaker is " + line[2:len(line)-2]
                    line = "then speaker want to " + line[2:len(line)-2]
                else:
                    line = "the " + str(relation_type) + " of this sentence is " + line[2:len(line)-2]
                text_a = texts[i] + line
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None))
        else:
            for i, line in enumerate(texts):
                guid = "%s-%s" % (set_type, i+1)
                examples.append(InputExample(guid=guid, text_a=line, text_b=None))
        return examples

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
        
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # if ex_index < 5:
        #     print("*** Example ***")
        #     print("guid: %s" % (example.guid))
        #     print("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     print(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()