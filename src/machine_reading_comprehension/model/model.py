import torch
import torch.nn as nn
import torch.nn.functional as F


from fastNLP.modules.aggregator.attention import BiAttention
# from fastNLP.modules import LSTM
from fastNLP.models import BaseModel
from fastNLP.core import Const
from fastNLP.core.utils import seq_len_to_mask
from fastNLP.modules.encoder import Embedding

from model.modules  import Linear, LSTM

class BiDAF(BaseModel):
    def __init__(self, char_vocab_size, init_embed=None,
                char_dim=8, char_channel_size=100, char_channel_width=5,
                word_dim=100, hidden_size=100, dropout=0.2):
        super(BiDAF, self).__init__()

        self.char_vocab_size = char_vocab_size
        self.char_dim = char_dim
        self.char_channel_size = char_channel_size
        self.char_channel_width = char_channel_width
        self.word_dim = word_dim
        self.hidden_size = hidden_size
        self.dropout = dropout

        # 1. Character Embedding Layer
        self.char_emb = nn.Embedding(self.char_vocab_size, self.char_dim, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        self.char_conv = nn.Conv2d(1, self.char_channel_size, (self.char_dim, self.char_channel_width))

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        # self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)
        self.word_emb = Embedding(init_embed)
        # Embedding = self.embed = encoder.Embedding(init_embed)

        # highway network
        assert self.hidden_size * 2 == (self.char_channel_size + self.word_dim)
        for i in range(2):
            setattr(self, f'highway_linear{i}',
                    nn.Sequential(Linear(self.hidden_size * 2, self.hidden_size * 2),
                                  nn.ReLU()))
            setattr(self, f'highway_gate{i}',
                    nn.Sequential(Linear(self.hidden_size * 2, self.hidden_size * 2),
                                  nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=self.hidden_size * 2,
                                 hidden_size=self.hidden_size,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=self.dropout)

        # 4. Attention Flow Layer
        self.att_weight_c = Linear(self.hidden_size * 2, 1)
        self.att_weight_q = Linear(self.hidden_size * 2, 1)
        self.att_weight_cq = Linear(self.hidden_size * 2, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = LSTM(input_size=self.hidden_size * 8,
                                   hidden_size=self.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=self.dropout)

        self.modeling_LSTM2 = LSTM(input_size=self.hidden_size * 2,
                                   hidden_size=self.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=self.dropout)

        # 6. Output Layer
        self.p1_weight_g = Linear(self.hidden_size * 8, 1, dropout=self.dropout)
        self.p1_weight_m = Linear(self.hidden_size * 2, 1, dropout=self.dropout)
        self.p2_weight_g = Linear(self.hidden_size * 8, 1, dropout=self.dropout)
        self.p2_weight_m = Linear(self.hidden_size * 2, 1, dropout=self.dropout)

        self.output_LSTM = LSTM(input_size=self.hidden_size * 14,
                                hidden_size=self.hidden_size,
                                bidirectional=True,
                                batch_first=True,
                                dropout=self.dropout)

        self.dropout = nn.Dropout(p=self.dropout)

    def _masked_softmax(self, logits, lens):
        mask = self._length_to_mask(lens, max_len=logits.shape[1])
        return F.softmax(logits - (1.0 - mask).float() * 10e6 )

    def _length_to_mask(self, length, max_len=None, dtype=None):
        """length: B.
        return B x max_len.
        If max_len is None, then max of length will be used.
        """
        assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
        max_len = max_len or length.max().item()
        mask = torch.arange(max_len, device=length.device,
                            dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
        if dtype is not None:
            mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
        return mask

    def forward(self, context_char, context_word , context_word_len,
                question_char, question_word, question_word_len):
        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            x = self.dropout(self.char_emb(x))
            # (batch * seq_len, 1, char_dim, word_len)
            x = x.view(-1, self.char_dim, x.size(2)).unsqueeze(1)
            # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
            x = self.char_conv(x).squeeze(2)
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, self.char_channel_size)

            return x

        def highway_network(x1, x2):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, f'highway_linear{i}')(x)
                g = getattr(self, f'highway_gate{i}')(x)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        def att_flow_layer(c, q, c_ls, q_ls):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            cq = []
            for i in range(q_len):
                #(batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                #(batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            mask_c = self._length_to_mask(c_ls, c_len).reshape(-1, c_len, 1).expand(-1, -1 ,q_len)
            mask_q = self._length_to_mask(q_ls, q_len).reshape(-1, 1, q_len).expand(-1, c_len, -1)
            mask = mask_c * mask_q

            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq
            s =  s - (1 - mask).float() * 10e6

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze()
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def output_layer(g, m, l):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            bs = g.size(0)
            c_len = g.size(1)
            hs = m.size(2)
            # (batch, c_len)
            p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze()

            pred1 = self._masked_softmax(p1, context_word_len)
            
            start_repr = torch.sum(m * pred1.unsqueeze(2).expand(bs, c_len, hs), dim=1)
            tiled_start_repr = start_repr.unsqueeze(1).expand(bs, c_len, hs)

            m_ = torch.cat([g, m, tiled_start_repr, m * tiled_start_repr], dim=-1)

            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM((m_, l))[0]
            # (batch, c_len)
            p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze()

            return p1, p2

        # 1. Character Embedding Layer
        c_char = char_emb_layer(context_char)
        q_char = char_emb_layer(question_char)
        # 2. Word Embedding Layer
        c_word = self.word_emb(context_word)
        q_word = self.word_emb(question_word)
        c_lens = context_word_len
        q_lens = question_word_len

        # Highway network
        c = highway_network(c_char, c_word)
        q = highway_network(q_char, q_word)
        # 3. Contextual Embedding Layer
        c = self.context_LSTM((c, c_lens))[0]
        q = self.context_LSTM((q, q_lens))[0]
        # 4. Attention Flow Layer
        g = att_flow_layer(c, q, context_word_len, question_word_len)
        # 5. Modeling Layer
        m = self.modeling_LSTM2((self.modeling_LSTM1((g, c_lens))[0], c_lens))[0]
        # 6. Output Layer
        start_logits, end_logits = output_layer(g, m, c_lens)

        # (batch, c_len), (batch, c_len)
        return {'start_logits': start_logits, "end_logits": end_logits}

    def predict(self, context_char, context_word , context_word_len,
                question_char, question_word, question_word_len):

        output = self.forward(context_char, context_word , context_word_len,
                              question_char, question_word, question_word_len)
        pred1 = self._masked_softmax(output['start_logits'], context_word_len)
        pred2 = self._masked_softmax(output['end_logits'], context_word_len)

        # pred1 = torch.argmax(prob1, dim=1, keepdim=True)
        # pred2 = torch.argmax(prob2, dim=1, keepdim=True)
        return {'pred1': pred1, 'pred2': pred2}
