import sys
sys.path.append('/remote-home/competition/Bidaf/fastNLP')
sys.path.append('./utils')
import os
os.environ['FASTNLP_BASE_URL'] = 'http://10.141.222.118:8888/file/download/'
os.environ['FASTNLP_CACHE_DIR'] = './cache'

import pickle
from torch.optim import Adam

from model.model_2 import BiDAF
from model.loss import BidafLoss

from fastNLP import Trainer
from fastNLP import BucketSampler

from model.metric_cck_try1 import SQuADMetric
from fastNLP.modules.encoder.embedding import StaticEmbedding

from utils.squad import SquadDataset

device = 'cuda:0'
lr = 0.001

dev_file = 'cache/data/dev-v1.1.json'

with open("utils/squad_data.pkl", "rb") as f:
    dataset = pickle.load(f)
train_data = dataset.get_train_data()
dev_data = dataset.get_dev_data()
word_vocab = dataset.word_vocab
char_vocab = dataset.char_vocab

dev_context_word_field = dev_data.get_field('context_word')
dev_context_word = dev_context_word_field.content

embed = StaticEmbedding(word_vocab, model_dir_or_name="/remote-home/competition/Bidaf/fastNLP/reproduction/machine_reading_comprehension/cache/glove.6B.100d.txt", requires_grad=True)

# vocab中char vocab的size
char_vocab_size = len(char_vocab)
print("char_vocab_size: {}".format(char_vocab_size))

model = BiDAF(char_vocab_size=char_vocab_size, init_embed=embed)

# sampler = BucketSampler(batch_size=32)
optimizer = Adam(model.parameters(), lr=lr)
metric = SQuADMetric(right_open=False,max_answer_len=17,
                    dev_file=dev_file,dev_context_word=dev_context_word,
                    word_vocab=word_vocab)
trainer = Trainer(train_data, model, optimizer=optimizer, loss=BidafLoss(padding_idx=0), batch_size=60,
                  n_epochs=15, print_every=50, dev_data=dev_data, metrics=metric, metric_key="f1",
                  validate_every=1000, save_path='./tmp_model_ori/', use_tqdm=True, device=device,
                  check_code_level=0)
trainer.train()
