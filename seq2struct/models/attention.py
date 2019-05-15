import attr
import numpy as np
import torch

from seq2struct import batching
from seq2struct.utils import registry
from seq2struct.models import transformer


def maybe_mask(attn, attn_mask):
    if attn_mask is not None:
        assert all(
            a == 1 or b == 1 or a == b
             for a, b in zip(attn.shape[::-1], attn_mask.shape[::-1])), \
            'Attention mask shape {} should be broadcastable with attention shape {}'.format(
                attn_mask.shape, attn.shape)

        attn.data.masked_fill_(attn_mask, -float('inf'))


@attr.s(frozen=True)
class AttentionBatchKey(batching.BatchKey):
    ref_path = attr.ib()

    @property
    def iterable_keys(self):
        return [0, 1]

    def call_batched(self, existing_results, callable, query, values, attn_mask=None):
        # query: list of tensors with shape [1, query_size]
        # values: list of tensors with shape [1, num values, value_size]
        batch_size = len(query)
        num_values = [v.shape[1] for v in values]
        max_num_values = max(v.shape[1] for v in values)

        batched_query = torch.cat(query, dim=0)
        batched_values = values[0].new(
            batch_size, max_num_values, values[0].shape[2]).fill_(0)
        for i, v in enumerate(values):
            batched_values[i:i+1] = v
        
        # TODO: Handle other cases later
        assert attn_mask is None or all(m is None for m in attn_mask)
        ranges = torch.arange(0, max_num_values).unsqueeze(0).expand(batch_size, -1)
        attn_mask = (ranges >= torch.tensor(num_values).unsqueeze(1)).to(values[0].device)

        output, attn = callable(query, values, attn_mask)
        return torch.split(output, dim=0), torch.split(attn, dim=0)
    


class Attention(torch.nn.Module):

    class BatchCollator(batching.BatchCollator):
        @classmethod
        def batch_key(cls, orig_type, ref_path, query, values, attn_mask=None):
            return AttentionBatchKey(ref_path=ref_path)

    def __init__(self, pointer):
        super().__init__()
        self.pointer = pointer
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query, values, attn_mask=None):
        # query shape: batch x query_size
        # values shape: batch x num values x value_size

        # attn_logits shape: batch x num values
        attn_logits = self.pointer(query, values, attn_mask)
        # attn_logits shape: batch x num values
        attn = self.softmax(attn_logits)
        # output shape: batch x 1 x value_size
        output = torch.bmm(attn.unsqueeze(1), values)
        output = output.squeeze(1)
        return output, attn


@registry.register('pointer', 'sdp')
class ScaledDotProductPointer(torch.nn.Module):
    def __init__(self, query_size, key_size):
        super().__init__()
        self.query_proj = torch.nn.Linear(query_size, key_size)
        self.temp = np.power(key_size, 0.5)
    
    def forward(self, query, keys, attn_mask=None):
        # query shape: batch x query_size
        # keys shape: batch x num keys x key_size

        # proj_query shape: batch x key_size x 1
        proj_query = self.query_proj(query).unsqueeze(2)
        
        # attn_logits shape: batch x num keys
        attn_logits = torch.bmm(keys, proj_query).squeeze(2) / self.temp
        maybe_mask(attn_logits, attn_mask)
        return attn_logits


@registry.register('attention', 'sdp')
class ScaledDotProductAttention(Attention):
    def __init__(self, query_size, value_size):
        super().__init__(ScaledDotProductPointer(query_size, value_size))


@registry.register('pointer', 'bahdanau')
class BahdanauPointer(torch.nn.Module):
    def __init__(self, query_size, key_size, proj_size):
        super().__init__()
        self.compute_scores = torch.nn.Sequential(
            torch.nn.Linear(query_size + key_size, proj_size),
            torch.nn.Tanh(),
            torch.nn.Linear(proj_size, 1))
    
    def forward(self, query: torch.Tensor, keys: torch.Tensor, attn_mask=None):
        # query shape: batch x query_size
        # keys shape: batch x num keys x key_size

        # query_expanded shape: batch x num keys x query_size
        query_expanded = query.unsqueeze(1).expand(-1, keys.shape[1], -1)

        # scores shape: batch x num keys x 1
        attn_logits = self.compute_scores(
            # shape: batch x num keys x query_size + key_size
            torch.cat((query_expanded, keys),
            dim=2))
        # scores shape: batch x num keys
        attn_logits = attn_logits.squeeze(2)
        maybe_mask(attn_logits, attn_mask)
        return attn_logits


@registry.register('attention', 'bahdanau')
class BahdanauAttention(Attention):
    def __init__(self, query_size, value_size, proj_size):
        super().__init__(BahdanauPointer(query_size, value_size, proj_size))


# Adapted from The Annotated Transformers
class MultiHeadedAttention(torch.nn.Module):

    BatchCollator = Attention.BatchCollator

    def __init__(self, h, query_size, value_size, dropout=0.1):
        super().__init__()
        assert query_size % h == 0
        assert value_size % h == 0

        # We assume d_v always equals d_k
        self.d_k = value_size // h
        self.h = h

        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(query_size, value_size),
            torch.nn.Linear(value_size, value_size),
            torch.nn.Linear(value_size, value_size),
            torch.nn.Linear(value_size, value_size),
        ])

        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)
        
    def forward(self, query, values, attn_mask=None):
        "Implements Figure 2"
        if attn_mask is not None:
            # Same mask applied to all h heads.
            attn_mask = attn_mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, keys, values = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, values, values))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = transformer.attention(
                query, keys, values, mask=attn_mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        x = x.squeeze(1)
        return self.linears[-1](x), self.attn
