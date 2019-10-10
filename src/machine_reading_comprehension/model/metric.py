import torch,json
from collections import OrderedDict, Counter
from tqdm import tqdm

import re
import string
import fastNLP
from fastNLP import FieldArray
from fastNLP.core.metrics import MetricBase


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
            # assert len(self.id_list) == len(pred_answer)
            if (len(self.id_list) == len(pred_answer)):
                answer_dict = dict(zip(self.id_list, pred_answer))
            else:
                answer_dict = dict(zip(self.id_list[:len(pred_answer)], pred_answer))
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

class SQuADMetric(MetricBase):
    r"""
    别名：:class:`fastNLP.SQuADMetric` :class:`fastNLP.core.metrics.SQuADMetric`

    SQuAD数据集metric
    
    :param pred1: 参数映射表中 `pred1` 的映射关系，None表示映射关系为 `pred1` -> `pred1`
    :param pred2: 参数映射表中 `pred2` 的映射关系，None表示映射关系为 `pred2` -> `pred2`
    :param target1: 参数映射表中 `target1` 的映射关系，None表示映射关系为 `target1` -> `target1`
    :param target2: 参数映射表中 `target2` 的映射关系，None表示映射关系为 `target2` -> `target2`
    :param float beta: f_beta分数， :math:`f_{beta} = \frac{(1 + {beta}^{2})*(pre*rec)}{({beta}^{2}*pre + rec)}` .
        常用为beta=0.5, 1, 2. 若为0.5则精确率的权重高于召回率；若为1，则两者平等；若为2，则召回率权重高于精确率。
    :param bool right_open: right_open为true表示start跟end指针指向一个左闭右开区间，为false表示指向一个左闭右闭区间。
    :param bool print_predict_stat: True则输出预测答案是否为空与正确答案是否为空的统计信息, False则不输出
    
    """
    
    def __init__(self, right_open=True, max_answer_len=17,dev_file=None,dev_context_word=None,word_vocab=None):
        
        super(SQuADMetric, self).__init__()
        self.right_open = right_open
        self.max_answer_len = max_answer_len
        self.dev_file = dev_file
        self.evaluator = SquadEvaluator(file_path = dev_file)
        self.answers = []
        self.dev_context_word = dev_context_word
        if isinstance(dev_context_word,FieldArray):
            print("Got Fastnlp's FildArray")
            self.dev_context_word = self.dev_context_word.content
        self.word_vocab = word_vocab

    def evaluate(self, pred1, pred2, target1, target2):
        """evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param pred1: [batch]或者[batch, seq_len], 预测答案开始的index, 如果SQuAD2.0中答案为空则为0
        :param pred2: [batch]或者[batch, seq_len] 预测答案结束的index, 如果SQuAD2.0中答案为空则为-1(左闭右闭区间)或者0(左闭右开区间)
        :param target1: [batch], 正确答案开始的index, 如果SQuAD2.0中答案为空则为0
        :param target2: [batch], 正确答案结束的index, 如果SQuAD2.0中答案为空则为-1(左闭右闭区间)或者0(左闭右开区间)
        :return: None
        """
        start, end = self._get_best_answer(pred1, pred2)
        self.answers += [x for x in zip(start,end)]
        
    
    def get_metric(self, reset=True):
        """get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果."""

        pred_answer = []
        print("Number of answers:{}".format(len(self.answers)))
        print("Size of context_word:{}".format(len(self.dev_context_word)))
        for idx,span in enumerate(self.answers):
            if idx >= len(self.dev_context_word):
                print("Attention size at :{}/{}".format(idx,len(self.answers)))
                break
            c_word = self.dev_context_word[idx]
            answer = c_word[span[0]:span[1]+1]
            answer = " ".join([self.word_vocab.to_word(w) for w in answer])
            pred_answer.append(answer)
        socre = self.evaluator.get_score(pred_answer)
        if reset:
            self.answers = []
        return socre

    def _get_best_answer(self, pred1, pred2):
        start = []
        end = []
        pred1 = pred1.cpu().tolist()
        pred2 = pred2.cpu().tolist()
        for i in range(len(pred1)):
            max_prob, max_start, max_end = 0, 0, 0
            for e in range(len(pred2[i])):
                for s in range(max(0, e - self.max_answer_len + 1), e + 1):
                    prob = pred1[i][s] * pred2[i][e]
                    if prob > max_prob:
                        max_start, max_end = s, e
                        max_prob = prob
            start.append(max_start)
            end.append(max_end)
        return start, end

class SQuADMetric_v00(MetricBase):
    r"""
    别名：:class:`fastNLP.SQuADMetric` :class:`fastNLP.core.metrics.SQuADMetric`

    SQuAD数据集metric
    
    :param pred1: 参数映射表中 `pred1` 的映射关系，None表示映射关系为 `pred1` -> `pred1`
    :param pred2: 参数映射表中 `pred2` 的映射关系，None表示映射关系为 `pred2` -> `pred2`
    :param target1: 参数映射表中 `target1` 的映射关系，None表示映射关系为 `target1` -> `target1`
    :param target2: 参数映射表中 `target2` 的映射关系，None表示映射关系为 `target2` -> `target2`
    :param float beta: f_beta分数， :math:`f_{beta} = \frac{(1 + {beta}^{2})*(pre*rec)}{({beta}^{2}*pre + rec)}` .
        常用为beta=0.5, 1, 2. 若为0.5则精确率的权重高于召回率；若为1，则两者平等；若为2，则召回率权重高于精确率。
    :param bool right_open: right_open为true表示start跟end指针指向一个左闭右开区间，为false表示指向一个左闭右闭区间。
    :param bool print_predict_stat: True则输出预测答案是否为空与正确答案是否为空的统计信息, False则不输出
    
    """
    
    def __init__(self, pred1=None, pred2=None, target1=None, target2=None,
                 beta=1, right_open=True, print_predict_stat=False):
        
        super(SQuADMetric_v00, self).__init__()
        
        self._init_param_map(pred1=pred1, pred2=pred2, target1=target1, target2=target2)
        
        self.print_predict_stat = print_predict_stat
        
        self.no_ans_correct = 0
        self.no_ans_wrong = 0
        
        self.has_ans_correct = 0
        self.has_ans_wrong = 0
        
        self.has_ans_f = 0.
        
        self.no2no = 0
        self.no2yes = 0
        self.yes2no = 0
        self.yes2yes = 0
        
        self.f_beta = beta
        
        self.right_open = right_open
    
    def evaluate(self, pred1, pred2, target1, target2):
        """evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param pred1: [batch]或者[batch, seq_len], 预测答案开始的index, 如果SQuAD2.0中答案为空则为0
        :param pred2: [batch]或者[batch, seq_len] 预测答案结束的index, 如果SQuAD2.0中答案为空则为-1(左闭右闭区间)或者0(左闭右开区间)
        :param target1: [batch], 正确答案开始的index, 如果SQuAD2.0中答案为空则为0
        :param target2: [batch], 正确答案结束的index, 如果SQuAD2.0中答案为空则为-1(左闭右闭区间)或者0(左闭右开区间)
        :return: None
        """
        pred_start = pred1
        pred_end = pred2
        target_start = target1
        target_end = target2
        
        if len(pred_start.size()) == 2:
            start_inference = pred_start.max(dim=-1)[1].cpu().tolist()
        else:
            start_inference = pred_start.cpu().tolist()
        if len(pred_end.size()) == 2:
            end_inference = pred_end.max(dim=-1)[1].cpu().tolist()
        else:
            end_inference = pred_end.cpu().tolist()
        
        start, end = [], []
        max_len = pred_start.size(1)
        t_start = target_start.cpu().tolist()
        t_end = target_end.cpu().tolist()
        
        for s, e in zip(start_inference, end_inference):
            start.append(min(s, e))
            end.append(max(s, e))
        for s, e, ts, te in zip(start, end, t_start, t_end):
            if not self.right_open:
                e += 1
                te += 1
            if ts == 0 and te == 0:
                if s == 0 and e == 0:
                    self.no_ans_correct += 1
                    self.no2no += 1
                else:
                    self.no_ans_wrong += 1
                    self.no2yes += 1
            else:
                if s == 0 and e == 0:
                    self.yes2no += 1
                else:
                    self.yes2yes += 1
                
                if s == ts and e == te:
                    self.has_ans_correct += 1
                else:
                    self.has_ans_wrong += 1
                a = [0] * s + [1] * (e - s) + [0] * (max_len - e)
                b = [0] * ts + [1] * (te - ts) + [0] * (max_len - te)
                a, b = torch.tensor(a), torch.tensor(b)
                
                TP = int(torch.sum(a * b))
                pre = TP / int(torch.sum(a)) if int(torch.sum(a)) > 0 else 0
                rec = TP / int(torch.sum(b)) if int(torch.sum(b)) > 0 else 0
                
                if pre + rec > 0:
                    f = (1 + (self.f_beta ** 2)) * pre * rec / ((self.f_beta ** 2) * pre + rec)
                else:
                    f = 0
                self.has_ans_f += f
    
    def get_metric(self, reset=True):
        """get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果."""
        evaluate_result = {}
        
        if self.no_ans_correct + self.no_ans_wrong + self.has_ans_correct + self.has_ans_wrong <= 0:
            return evaluate_result
        
        evaluate_result['EM'] = 0
        evaluate_result[f'f_{self.f_beta}'] = 0
        
        flag = 0
        
        if self.no_ans_correct + self.no_ans_wrong > 0:
            evaluate_result[f'noAns-f_{self.f_beta}'] = \
                round(100 * self.no_ans_correct / (self.no_ans_correct + self.no_ans_wrong), 3)
            evaluate_result['noAns-EM'] = \
                round(100 * self.no_ans_correct / (self.no_ans_correct + self.no_ans_wrong), 3)
            evaluate_result[f'f_{self.f_beta}'] += evaluate_result[f'noAns-f_{self.f_beta}']
            evaluate_result['EM'] += evaluate_result['noAns-EM']
            flag += 1
        
        if self.has_ans_correct + self.has_ans_wrong > 0:
            evaluate_result[f'hasAns-f_{self.f_beta}'] = \
                round(100 * self.has_ans_f / (self.has_ans_correct + self.has_ans_wrong), 3)
            evaluate_result['hasAns-EM'] = \
                round(100 * self.has_ans_correct / (self.has_ans_correct + self.has_ans_wrong), 3)
            evaluate_result[f'f_{self.f_beta}'] += evaluate_result[f'hasAns-f_{self.f_beta}']
            evaluate_result['EM'] += evaluate_result['hasAns-EM']
            flag += 1
        
        if self.print_predict_stat:
            evaluate_result['no2no'] = self.no2no
            evaluate_result['no2yes'] = self.no2yes
            evaluate_result['yes2no'] = self.yes2no
            evaluate_result['yes2yes'] = self.yes2yes
        
        if flag <= 0:
            return evaluate_result

        
        evaluate_result[f'f_{self.f_beta}'] = round(evaluate_result[f'f_{self.f_beta}'] / flag, 3)
        evaluate_result['EM'] = round(evaluate_result['EM'] / flag, 3)
        
        if reset:
            self.no_ans_correct = 0
            self.no_ans_wrong = 0
            
            self.has_ans_correct = 0
            self.has_ans_wrong = 0
            
            self.has_ans_f = 0.
            
            self.no2no = 0
            self.no2yes = 0
            self.yes2no = 0
            self.yes2yes = 0
        
        return evaluate_result

    
class SQuADMetric_v01(MetricBase):
    r"""
    别名：:class:`fastNLP.SQuADMetric` :class:`fastNLP.core.metrics.SQuADMetric`

    SQuAD数据集metric
    
    :param pred1: 参数映射表中 `pred1` 的映射关系，None表示映射关系为 `pred1` -> `pred1`
    :param pred2: 参数映射表中 `pred2` 的映射关系，None表示映射关系为 `pred2` -> `pred2`
    :param target1: 参数映射表中 `target1` 的映射关系，None表示映射关系为 `target1` -> `target1`
    :param target2: 参数映射表中 `target2` 的映射关系，None表示映射关系为 `target2` -> `target2`
    :param float beta: f_beta分数， :math:`f_{beta} = \frac{(1 + {beta}^{2})*(pre*rec)}{({beta}^{2}*pre + rec)}` .
        常用为beta=0.5, 1, 2. 若为0.5则精确率的权重高于召回率；若为1，则两者平等；若为2，则召回率权重高于精确率。
    :param bool right_open: right_open为true表示start跟end指针指向一个左闭右开区间，为false表示指向一个左闭右闭区间。
    :param bool print_predict_stat: True则输出预测答案是否为空与正确答案是否为空的统计信息, False则不输出
    
    """
    
    def __init__(self, pred1=None, pred2=None, target1=None, target2=None,
                 beta=1, right_open=True, print_predict_stat=False, max_answer_len=17):
        
        super(SQuADMetric_v01, self).__init__()
        
        self._init_param_map(pred1=pred1, pred2=pred2, target1=target1, target2=target2)
        
        self.print_predict_stat = print_predict_stat

        self.max_answer_len = max_answer_len
        
        self.no_ans_correct = 0
        self.no_ans_wrong = 0
        
        self.has_ans_correct = 0
        self.has_ans_wrong = 0
        
        self.has_ans_f = 0.
        
        self.no2no = 0
        self.no2yes = 0
        self.yes2no = 0
        self.yes2yes = 0
        
        self.f_beta = beta
        
        self.right_open = right_open
    
    def evaluate(self, pred1, pred2, target1, target2):
        """evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param pred1: [batch]或者[batch, seq_len], 预测答案开始的index, 如果SQuAD2.0中答案为空则为0
        :param pred2: [batch]或者[batch, seq_len] 预测答案结束的index, 如果SQuAD2.0中答案为空则为-1(左闭右闭区间)或者0(左闭右开区间)
        :param target1: [batch], 正确答案开始的index, 如果SQuAD2.0中答案为空则为0
        :param target2: [batch], 正确答案结束的index, 如果SQuAD2.0中答案为空则为-1(左闭右闭区间)或者0(左闭右开区间)
        :return: None
        """
        pred_start = pred1
        pred_end = pred2
        target_start = target1
        target_end = target2
        
        # if len(pred_start.size()) == 2:
        #     start_inference = pred_start.max(dim=-1)[1].cpu().tolist()
        # else:
        #     start_inference = pred_start.cpu().tolist()
        # if len(pred_end.size()) == 2:
        #     end_inference = pred_end.max(dim=-1)[1].cpu().tolist()
        # else:
        #     end_inference = pred_end.cpu().tolist()
        
        # start, end = [], []

        start, end = self._get_best_answer(pred1, pred2)
        

        max_len = pred_start.size(1)
        t_start = target_start.cpu().tolist()
        t_end = target_end.cpu().tolist()
        
        # for s, e in zip(start_inference, end_inference):
        #     start.append(min(s, e))
        #     end.append(max(s, e))
        for s, e, ts, te in zip(start, end, t_start, t_end):
            if not self.right_open:
                e += 1
                te += 1
            if ts == 0 and te == 0:
                if s == 0 and e == 0:
                    self.no_ans_correct += 1
                    self.no2no += 1
                else:
                    self.no_ans_wrong += 1
                    self.no2yes += 1
            else:
                if s == 0 and e == 0:
                    self.yes2no += 1
                else:
                    self.yes2yes += 1
                
                if s == ts and e == te:
                    self.has_ans_correct += 1
                else:
                    self.has_ans_wrong += 1
                a = [0] * s + [1] * (e - s) + [0] * (max_len - e)
                b = [0] * ts + [1] * (te - ts) + [0] * (max_len - te)
                a, b = torch.tensor(a), torch.tensor(b)
                
                TP = int(torch.sum(a * b))
                pre = TP / int(torch.sum(a)) if int(torch.sum(a)) > 0 else 0
                rec = TP / int(torch.sum(b)) if int(torch.sum(b)) > 0 else 0
                
                if pre + rec > 0:
                    f = (1 + (self.f_beta ** 2)) * pre * rec / ((self.f_beta ** 2) * pre + rec)
                else:
                    f = 0
                self.has_ans_f += f
    
    def get_metric(self, reset=True):
        """get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果."""
        evaluate_result = {}
        
        if self.no_ans_correct + self.no_ans_wrong + self.has_ans_correct + self.has_ans_wrong <= 0:
            return evaluate_result
        
        evaluate_result['EM'] = 0
        evaluate_result[f'f_{self.f_beta}'] = 0
        
        flag = 0
        
        if self.no_ans_correct + self.no_ans_wrong > 0:
            evaluate_result[f'noAns-f_{self.f_beta}'] = \
                round(100 * self.no_ans_correct / (self.no_ans_correct + self.no_ans_wrong), 3)
            evaluate_result['noAns-EM'] = \
                round(100 * self.no_ans_correct / (self.no_ans_correct + self.no_ans_wrong), 3)
            evaluate_result[f'f_{self.f_beta}'] += evaluate_result[f'noAns-f_{self.f_beta}']
            evaluate_result['EM'] += evaluate_result['noAns-EM']
            flag += 1
        
        if self.has_ans_correct + self.has_ans_wrong > 0:
            evaluate_result[f'hasAns-f_{self.f_beta}'] = \
                round(100 * self.has_ans_f / (self.has_ans_correct + self.has_ans_wrong), 3)
            evaluate_result['hasAns-EM'] = \
                round(100 * self.has_ans_correct / (self.has_ans_correct + self.has_ans_wrong), 3)
            evaluate_result[f'f_{self.f_beta}'] += evaluate_result[f'hasAns-f_{self.f_beta}']
            evaluate_result['EM'] += evaluate_result['hasAns-EM']
            flag += 1
        
        if self.print_predict_stat:
            evaluate_result['no2no'] = self.no2no
            evaluate_result['no2yes'] = self.no2yes
            evaluate_result['yes2no'] = self.yes2no
            evaluate_result['yes2yes'] = self.yes2yes
        
        if flag <= 0:
            return evaluate_result
        
        evaluate_result[f'f_{self.f_beta}'] = round(evaluate_result[f'f_{self.f_beta}'] / flag, 3)
        evaluate_result['EM'] = round(evaluate_result['EM'] / flag, 3)
        
        if reset:
            self.no_ans_correct = 0
            self.no_ans_wrong = 0
            
            self.has_ans_correct = 0
            self.has_ans_wrong = 0
            
            self.has_ans_f = 0.
            
            self.no2no = 0
            self.no2yes = 0
            self.yes2no = 0
            self.yes2yes = 0
        
        return evaluate_result

    def _get_best_answer(self, pred1, pred2):
        start = []
        end = []
        pred1 = pred1.cpu().tolist()
        pred2 = pred2.cpu().tolist()
        for i in range(len(pred1)):
            max_prob, max_start, max_end = 0, 0, 0
            for e in range(len(pred2[i])):
                for s in range(max(0, e - self.max_answer_len + 1), e + 1):
                    prob = pred1[i][s] * pred2[i][e]
                    if prob > max_prob:
                        max_start, max_end = s, e
                        max_prob = prob
            start.append(max_start)
            end.append(max_end)
        return start, end
    
