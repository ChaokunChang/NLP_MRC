import torch
import torch.nn.functional as F

from fastNLP.core.losses import LossBase
from fastNLP.core.utils import seq_len_to_mask

class BidafLoss(LossBase):
    def __init__(self, start_logits=None, end_logits=None, target1=None, target2=None,
                seq_len=None, padding_idx=-100):
        super(BidafLoss, self).__init__()
        self._init_param_map(start_logits=start_logits, end_logits=end_logits,
                            target1=target1, target2=target2, seq_len=seq_len)
        self.padding_idx = padding_idx
    
    def get_loss(self, start_logits, end_logits, target1, target2, seq_len=None):
        # if seq_len is not None:
        #     mask = seq_len_to_mask(seq_len).view(-1).eq(0)
        #     target1 = target1.masked_fill(mask, self.padding_idx)
        #     target2 = target2.masked_fill(mask, self.padding_idx)

        return F.cross_entropy(input=start_logits, target=target1, ignore_index=self.padding_idx) + \
            F.cross_entropy(input=end_logits, target=target2, ignore_index=self.padding_idx)
        # if pred.dim()>2:
        #     pred = pred.view(-1, pred.size(-1))
        #     target = target.view(-1)

        # return F.cross_entropy(input=pred, target=target,
        #                        ignore_index=self.padding_idx)