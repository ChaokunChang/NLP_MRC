import sys
sys.path.append('/remote-home/competition/Bidaf/fastNLP')
sys.path.append('./utils')
import os
os.environ['FASTNLP_BASE_URL'] = 'http://10.141.222.118:8888/file/download/'
os.environ['FASTNLP_CACHE_DIR'] = './cache'

import pickle
import torch
from torch.optim import Adam
from model.model import BiDAF
from model.loss import BidafLoss
from utils.squad import SquadEvaluator

# from fastNLP.io.embed_loader import EmbeddingOption
# from fastNLP.core.vocabulary import VocabularyOption
from fastNLP import Trainer,DataSetIter
from fastNLP import BucketSampler,SequentialSampler
# from fastNLP import GradientClipCallback
from fastNLP.modules.encoder.embedding import StaticEmbedding
from utils.squad import SquadDataset

max_answer_len = 17
def get_best_answer(pred1, pred2):
    start = []
    end = []
    pred1 = pred1.cpu().tolist()
    pred2 = pred2.cpu().tolist()
    for i in range(len(pred1)):
        max_prob, max_start, max_end = 0, 0, 0
        for e in range(len(pred2[i])):
            for s in range(max(0, e - max_answer_len + 1), e + 1):
                prob = pred1[i][s] * pred2[i][e]
                if prob > max_prob:
                    max_start, max_end = s, e
                    max_prob = prob
        start.append(max_start)
        end.append(max_end)
    return start, end

device = 'cuda:0'
lr = 0.1
with open("utils/squad_data.pkl", "rb") as f:
    dataset = pickle.load(f)

dev_data = dataset.get_dev_data()
word_vocab = dataset.word_vocab
dev_context_word_field = dev_data.get_field('context_word')
dev_context_word = dev_context_word_field.content

model_path = "/remote-home/competition/Bidaf/fastNLP/reproduction/machine_reading_comprehension/tmp_1/best_BiDAF_f_1_2019-06-29-05-16-19"
print("Loading model from {}".format(model_path))
dev_file = 'cache/data/dev-v1.1.json'
model = torch.load(model_path)
model.eval()
evaluator = SquadEvaluator(dev_file)
batch_size = 256
dev_iter = DataSetIter(dataset=dev_data,batch_size=batch_size,sampler=SequentialSampler())
results = []
processed_num = 0
for batch_x,batch_y in dev_iter:
    print("Batch shape:{}".format(batch_x['context_char'].shape))
    ans = model(batch_x['context_char'],batch_x['context_word'],batch_x['context_word_len'],
                batch_x['question_char'],batch_x['question_word'],batch_x['question_word_len'])

    pred1,pred2 = get_best_answer(ans['start_logits'],ans['end_logits'])
    ans = [x for x in zip(pred1,pred2)]
    results += ans
    processed_num += batch_size
    print("Predicted {} records.".format(processed_num))

with open('./tmp_results.pkl','wb') as f:
    pickle.dump(results,f)

pred_answer = []
for idx,span in enumerate(results):
    c_word = dev_context_word[idx]
    answer = c_word[span[0]:span[1]+1]
    answer = " ".join([word_vocab.to_word(w) for w in answer])
    pred_answer.append(answer)
socre = evaluator.get_score(pred_answer)
print(socre)
