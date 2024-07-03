import os
import csv
import sys
import logging
import spacy
import numpy as np

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

        train_examples = processor.get_examples(base_attrs['data_path'], 'train')
        train_feats = self._get_bart_feats(args, train_examples, base_attrs)

        dev_examples = processor.get_examples(base_attrs['data_path'], 'dev')
        dev_feats = self._get_bart_feats(args, dev_examples, base_attrs)

        test_examples = processor.get_examples(base_attrs['data_path'], 'test')
        test_feats = self._get_bart_feats(args, test_examples, base_attrs)
        
        self.logger.info('Generate Text Features Finished...')

        return {
            'train': train_feats,
            'dev': dev_feats,
            'test': test_feats
        }

    def _get_bart_feats(self, args, examples, base_attrs):

        max_seq_length = base_attrs["benchmarks"]['max_seq_lengths']['text']

        tokenizer = base_attrs["tokenizer"]

        features = convert_examples_to_features(examples, max_seq_length, tokenizer)     
        features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids, feat.noun_mask] for feat in features]

        return features_list

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, nouns):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            nouns: list. The noun phrases in sentence text_a.
            Only must be specified for sequence pair tasks.
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.nouns = nouns

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, noun_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.noun_mask = noun_mask

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
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'dev':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_noun(self, nlp, sentence, noun_list):
        sentence_split = sentence.split(' ')
        pos_doc = [nlp(x) for x in sentence_split]

        nouns = []

        for i in range(len(pos_doc)):
            for j in range(len(pos_doc[i])):
                if pos_doc[i][j].tag_ in noun_list:
                    nouns.append(str(pos_doc[i][j]))

        # for i, split in enumerate(sentence_split):
        #     if len(split) == len(pos_doc[i]):
        #         noun_position = np.arange(0, len(split)).tolist()
        #         noun_positions.append(noun_position)
        #     else:
        #         for pos, word in enumerate(split):
        #             noun_position = []
        #             if word in pos_doc[i]:
        #                 noun_position.append(pos)
        #         noun_positions.append(noun_position)

        # for i, split in enumerate(sentence_split):
        #     if len(sentence_split[i]) != len(pos_doc[i]):
        #         new_sentence = []
        #         for token in pos_doc[i]:
        #             new_sentence.append(str(token))
        #         sentence_split[i] = new_sentence
        #         assert len(sentence_split[i]) == len(pos_doc[i])
        #     noun_position = []
        #     for j in range(len(pos_doc[i])):
        #         if pos_doc[i][j].tag_ in noun_list:
        #             noun_position.append(j)
        #     noun_positions.append(noun_position)
        # print(sentence_split, nouns)
        return nouns

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        nlp = spacy.load('en_core_web_sm')
        noun_list = ['NNP', 'NNPS', 'NN', 'NNS']

        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s_%s_%s_%s" % (set_type, line[0], line[1], line[2])
            text_a = line[3]

            nouns = self.get_noun(nlp, text_a, noun_list)

            examples.append(
                InputExample(guid=guid, text_a=text_a, nouns=nouns))
        return examples

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
        
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        # tokens_b = None
        # if example.text_b:
        #     tokens_b = tokenizer.tokenize(example.text_b)
        #     # Modifies `tokens_a` and `tokens_b` in place so that the total
        #     # length is less than the specified length.
        #     # Account for [CLS], [SEP], [SEP] with "- 3"
        #     _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        # else:
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

        # if tokens_b:
        #     tokens += tokens_b + ["[SEP]"]
        #     segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # 1 for noun tokens.
        noun_mask = []
        noun_ids = []
        for word in example.nouns:
            noun_ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)))
        
        for i in input_ids:
            if i in noun_ids:
                noun_mask += [1]
            else:
                noun_mask += [0]
        # for j, word in enumerate(example.text_a):
        #     bpes = tokenizer.tokenize(word, add_prefix_space=True)
        #     bpes = tokenizer.convert_tokens_to_ids(bpes)
        #     # print(word, bpes)
        #     if j in example.noun_positions[ex_index]:
        #         noun_mask += [1] * len(bpes)
        #     else:
        #         noun_mask += [0] * len(bpes)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # print(len(input_ids))
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        noun_mask += padding
        # print(noun_mask, input_ids, len(noun_mask), len(input_ids), max_seq_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(noun_mask) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          noun_mask=noun_mask)
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