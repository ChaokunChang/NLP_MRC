# coding: utf-8

# from utils.tokenizer import SpacyTokenizer
from .tokenizer import SpacyTokenizer
import sys
sys.path.append('/remote-home/competition/Bidaf/fastNLP')

import json
from collections import OrderedDict, Counter
from tqdm import tqdm
import logging
import re
import string
import os
import pickle as pkl
import fastNLP
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import Const


class SquadDataset():
    """ The SquadDataset for fastnlp
    Attributes:
        min_word_count: the minimal count in word vocabulary.
        min_char_count: the minimal count in char vocabulary.
        train_file: the train data file path.
        dev_file: the dev data file path.
        train_data: the train dataset ,Class: DataSet() (of fastnlp).
        dev_data: the dev dataset ,Class: DataSet() (of fastnlp).
        word_vocab: the word based vocabulary.
        word_vocab_size: the size of word based vocabulary.
        char_vocab: the char based vocabulary.
        char_vocab_size: the size of the char based vocabulary.
     """

    def __init__(self, min_word_count=3, min_char_count=10,
                 train_file=None, dev_file=None):
        self.min_word_count = min_word_count
        self.min_char_count = min_char_count
        self.train_file = train_file
        self.dev_file = dev_file
        print("Loading Squad data.")
        if self.dev_file is not None:
            suffix = self.dev_file.split('.')[-1]
            if suffix == "pkl":
                pickle = True
            else:
                pickle = False
            self.dev_data = self.load_file(self.dev_file, pickle_file=pickle)
        if self.train_file is not None:
            suffix = self.train_file.split('.')[-1]
            if suffix == "pkl":
                pickle = True
            else:
                pickle = False
            self.train_data = self.load_file(
                self.train_file, pickle_file=pickle)
        print("Building word vocab.")
        self.word_vocab = Vocabulary(min_freq=self.min_word_count)
        (self.word_vocab).from_dataset(self.train_data,
                                       self.dev_data, field_name=['context_word', 'question_word'])
        self.word_vocab.index_dataset(
            self.train_data, self.dev_data, field_name='context_word')
        self.word_vocab.index_dataset(
            self.train_data, self.dev_data, field_name='question_word')
        self.word_vocab_size = len(self.word_vocab)

        print("Building char vocab.")
        self.char_vocab = Vocabulary(min_freq=self.min_char_count)
        (self.char_vocab).from_dataset(self.train_data,
                                       self.dev_data, field_name=['context_char', 'question_char'])
        self.char_vocab.index_dataset(
            self.train_data, self.dev_data, field_name='question_char')
        self.char_vocab.index_dataset(
            self.train_data, self.dev_data, field_name='context_char')
        self.char_vocab_size = len(self.char_vocab)

    def load_file(self, file, pickle_file=True):
        if not pickle_file:
            reader = SquadReader()
            read_data = reader.read(file)
            s_file_prefix = file.split('/')[-1].split('.')[0]
            print(s_file_prefix)
            with open('./tmpreader.pkl', 'wb') as f:
                pkl.dump(read_data, f)
        else:
            with open(file, 'rb') as f:
                read_data = pkl.load(f)
        
        context = [x['context'] for x in read_data ]
        question = [x['question'] for x in read_data ]
        qid = [x['qid'] for x in read_data ]
        answer = [x['answer'] for x in read_data ]

        target1 = [x['answer_start'] for x in read_data]
        target2 = [x['answer_end'] for x in read_data]
        context_word = [x['context_word'] for x in read_data]
        context_word_len = [len(x['context_word']) for x in read_data]
        context_char = [ [ [c for c in w] for w in x['context_word']] for x in read_data]

        question_word = [x['question_word'] for x in read_data]
        question_word_len = [len(x['question_word']) for x in read_data]
        question_char = [[ [c for c in w] for w in x['question_word']] for x in read_data]

        data = DataSet({"context": context,"question": question,'qid': qid,
                        "answer": answer,
                        "target1": target1,
                        "target2": target2,
                        "context_char": context_char,
                        "context_word": context_word,
                        "context_word_len": context_word_len,
                        "question_char": question_char,
                        "question_word": question_word,
                        "question_word_len": question_word_len})

        data.set_input("context_char", "context_word", "context_word_len",
                       "question_char", "question_word", "question_word_len")
        data.set_target('target1', 'target2')
        print(type(data), len(data))
        return data

    def get_train_data(self):
        return self.train_data

    def get_dev_data(self):
        return self.dev_data

    def get_word_vocab(self):
        return self.word_vocab

    def get_char_vocab(self):
        return self.char_vocab


class SquadReader():
    def __init__(self, fine_grained=False):
        self.tokenizer = SpacyTokenizer(fine_grained)

    def read(self, file_path):
        instances = self._read(file_path)
        instances = [instance for instance in tqdm(instances)]
        return instances

    def _read(self, file_path, context_limit=-1):
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        for article in dataset:
            for paragraph in article['paragraphs']:
                context = paragraph["context"]
                context_tokens, context_token_spans = self.tokenizer.word_tokenizer(
                    context)
                context_char = []
                for token in context_tokens:
                    for char in token:
                        context_char.append(char)

                for question_answer in paragraph['qas']:
                    question = question_answer["question"].strip()
                    question_tokens, _ = self.tokenizer.word_tokenizer(
                        question)
                    question_char = []
                    for token in question_tokens:
                        for char in token:
                            question_char.append(char)

                    answers, span_starts, span_ends = [], [], []
                    if "answers" in question_answer:
                        answers = [answer['text']
                                   for answer in question_answer['answers']]
                        span_starts = [answer['answer_start']
                                       for answer in question_answer['answers']]
                        span_ends = [start + len(answer)
                                     for start, answer in zip(span_starts, answers)]

                    answer_char_spans = zip(span_starts, span_ends) if len(
                        span_starts) > 0 and len(span_ends) > 0 else None
                    answers = answers if len(answers) > 0 else None
                    qid = question_answer['id']
                    instance = self._make_instance(context, context_tokens, context_token_spans, context_char,
                                                   question, question_tokens, question_char, answer_char_spans,
                                                   span_starts, span_ends, answers, qid)
                    if len(instance['context_word']) > context_limit and context_limit > 0:
                        if instance['answer_start'] > context_limit or instance['answer_end'] > context_limit:
                            continue
                        else:
                            instance['context_word'] = instance['context_word'][:context_limit]
                    yield instance

    def _make_instance( self, context, context_tokens, context_token_spans, context_char,
                        question, question_tokens, question_char,
                        answer_char_spans=None, span_starts=None, span_ends=None,
                        answers=None, qid=None):
        answer_token_starts, answer_token_ends = [], []
        if answers is not None:
            for answer_char_start, answer_char_end in answer_char_spans:
                answer_token_span = []
                for idx, span in enumerate(context_token_spans):
                    if not (answer_char_end <= span[0] or answer_char_start >= span[1]):
                        answer_token_span.append(idx)

                assert len(answer_token_span) > 0
                answer_token_starts.append(answer_token_span[0])
                answer_token_ends.append(answer_token_span[-1])

        return OrderedDict({
            "context": context,
            "context_word": context_tokens,
            "context_word_spans": context_token_spans,
            "context_char": context_char,
            "context_word_len": [len(word) for word in context_tokens],
            "context_char_len": [len(char) for char in context_char],
            "question_word_len": [len(word) for word in question_tokens],
            "question_char_len": [len(char) for char in question_char],
            "question": question,
            'qid': qid,
            "question_word": question_tokens,
            "question_char": question_char,
            "answer": answers[0] if answers is not None else None,
            "answer_start": answer_token_starts[0] if answers is not None else None,
            "answer_end": answer_token_ends[0] if answers is not None else None,
            "answer_char_start": span_starts[0] if answers is not None else None,
            "answer_char_end": span_ends[0] if answers is not None else None,
        })


class SquadEvaluator():
    def __init__(self, file_path, monitor='f1'):
        self.ground_dict = dict()
        self.id_list = []
        self.monitor = monitor

        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        for article in dataset:
            for paragraph in article['paragraphs']:
                for question_answer in paragraph['qas']:
                    id = question_answer["id"]
                    self.ground_dict[id] = [answer['text']
                                            for answer in question_answer['answers']]
                    self.id_list.append(id)

    def get_monitor(self):
        return self.monitor

    def get_score(self, pred_answer):
        if isinstance(pred_answer, list):
            assert len(self.id_list) == len(pred_answer)
            answer_dict = dict(zip(self.id_list, pred_answer))
        else:
            answer_dict = pred_answer

        f1 = exact_match = total = 0
        for key, value in answer_dict.items():
            total += 1
            ground_truths = self.ground_dict[key]
            prediction = value
            exact_match += SquadEvaluator.metric_max_over_ground_truths(
                SquadEvaluator.exact_match_score, prediction, ground_truths)
            f1 += SquadEvaluator.metric_max_over_ground_truths(
                SquadEvaluator.f1_score, prediction, ground_truths)
        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total
        return {'exact_match': exact_match, 'f1': f1}

    @staticmethod
    def normalize_answer(s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def exact_match_score(prediction, ground_truth):
        return (SquadEvaluator.normalize_answer(prediction) == SquadEvaluator.normalize_answer(ground_truth))

    @staticmethod
    def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    @staticmethod
    def f1_score(prediction, ground_truth):
        prediction_tokens = SquadEvaluator.normalize_answer(prediction).split()
        ground_truth_tokens = SquadEvaluator.normalize_answer(
            ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


if __name__ == "__main__":
    # data_folder = "../cache/data"
    # train_file = os.path.join(data_folder,'train-v1.1.json')
    # dev_file = os.path.join(data_folder,'dev-v1.1.json')
    # train_file = '../data/trainreader.pkl'
    # dev_file = '../data/devreader.pkl'
    train_file = './train_reader.pkl'
    dev_file = './dev_reader.pkl'
    dataset = SquadDataset(train_file=train_file, dev_file=dev_file)
    save_path = './squad_data_all.pkl'
    with open(save_path, 'wb') as f:
        pkl.dump(dataset, f)
    print("SquadDataSet object saved in {}".format(save_path))
    # with open('./data/suqad_data_3_10.pkl','rb') as f:
    #     dataset = pkl.load(f)
    train_data = dataset.get_train_data()
    dev_data = dataset.get_dev_data()
    print(len(train_data), train_data)
    print(len(dev_data), dev_data)
