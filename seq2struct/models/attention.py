import numpy as np
import torch

from seq2struct.utils import registry


def maybe_mask(attn, attn_mask):
    if attn_mask is not None:
        assert all(
            a == 1 or b == 1 or a == b
             for a, b in zip(attn.shape[::-1], attn_mask.shape[::-1])), \
            'Attention mask shape {} should be broadcastable with attention shape {}'.format(
                attn_mask.shape, attn.shape)

        attn.data.masked_fill_(attn_mask, -float('inf'))


class Attention(torch.nn.Module):
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