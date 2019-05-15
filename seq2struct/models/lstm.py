# coding=utf-8
# Derived from
#    https://github.com/pcyin/tranX/blob/4e391c63a75a112b3dc8a3b1dcc87058136db1e4/model/lstm.py
#    https://github.com/pytorch/pytorch/blob/v0.4.1/torch/nn/modules/rnn.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.autograd import Variable
from torch.nn import Parameter, init
from torch.nn._functions.rnn import variable_recurrent_factory, Recurrent, StackedRNN
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.utils.rnn import PackedSequence
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend


class DropoutCreator(nn.Module):
    def __init__(self, device, batch_size, dropout=0.):
        super().__init__()
        self._device = device
        self.batch_size = batch_size
        self.dropout = dropout

    def forward(self):
        if self.dropout:
            if self.training:
                input_dropout_mask = torch.bernoulli(
                    torch.full(
                        (batch_size, 4, self.input_size),
                        1 - self.dropout,
                        device=self._device))
                h_dropout_mask = torch.bernoulli(
                    torch.full(
                        (batch_size, 4, self.input_size),
                        1 - self.dropout,
                        device=self._device))
            else:
                input_dropout_mask = h_dropout_mask = [1. - self.dropout] * 4
        else:
            input_dropout_mask = h_dropout_mask = [1.] * 4
        
        return input_dropout_mask, h_dropout_mask


class RecurrentDropoutLSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(RecurrentDropoutLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.W_i = Parameter(torch.Tensor(hidden_size, input_size))
        self.U_i = Parameter(torch.Tensor(hidden_size, hidden_size))
        # self.b_i = Parameter(torch.Tensor(hidden_size))

        self.W_f = Parameter(torch.Tensor(hidden_size, input_size))
        self.U_f = Parameter(torch.Tensor(hidden_size, hidden_size))
        # self.b_f = Parameter(torch.Tensor(hidden_size))

        self.W_c = Parameter(torch.Tensor(hidden_size, input_size))
        self.U_c = Parameter(torch.Tensor(hidden_size, hidden_size))
        # self.b_c = Parameter(torch.Tensor(hidden_size))

        self.W_o = Parameter(torch.Tensor(hidden_size, input_size))
        self.U_o = Parameter(torch.Tensor(hidden_size, hidden_size))
        # self.b_o = Parameter(torch.Tensor(hidden_size))

        self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))

        self._input_dropout_mask = self._h_dropout_mask = None

        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.W_i)
        init.orthogonal_(self.U_i)
        init.orthogonal_(self.W_f)
        init.orthogonal_(self.U_f)
        init.orthogonal_(self.W_c)
        init.orthogonal_(self.U_c)
        init.orthogonal_(self.W_o)
        init.orthogonal_(self.U_o)
        self.bias_ih.data.fill_(0.)
        # forget gate set to 1.
        self.bias_ih.data[self.hidden_size:2 * self.hidden_size].fill_(1.)
        self.bias_hh.data.fill_(0.)

    def set_dropout_masks(self, batch_size):
        if self.dropout:
            if self.training:
                new_tensor = self.W_i.data.new
                self._input_dropout_mask = Variable(torch.bernoulli(
                    new_tensor(4, batch_size, self.input_size).fill_(1 - self.dropout)), requires_grad=False)
                self._h_dropout_mask = Variable(torch.bernoulli(
                    new_tensor(4, batch_size, self.hidden_size).fill_(1 - self.dropout)), requires_grad=False)
            else:
                self._input_dropout_mask = self._h_dropout_mask = [1. - self.dropout] * 4
        else:
            self._input_dropout_mask = self._h_dropout_mask = [1.] * 4

    def forward(self, input, hidden_state, input_dropout_mask=None, h_dropout_mask=None):
        def get_mask_slice(mask, idx):
            if isinstance(mask, list): return mask[idx]
            else: return mask[idx][:input.size(0)]
        def get_mask_slice2(mask, idx):
            if isinstance(mask, list): return mask[idx]
            else: return mask[:, idx]

        h_tm1, c_tm1 = hidden_state

        # if self._input_dropout_mask is None:
        #     self.set_dropout_masks(input.size(0))

        if input_dropout_mask is None:
            xi_t = F.linear(input * get_mask_slice(self._input_dropout_mask, 0), self.W_i)
            xf_t = F.linear(input * get_mask_slice(self._input_dropout_mask, 1), self.W_f)
            xc_t = F.linear(input * get_mask_slice(self._input_dropout_mask, 2), self.W_c)
            xo_t = F.linear(input * get_mask_slice(self._input_dropout_mask, 3), self.W_o)
        else:
            xi_t = F.linear(input * get_mask_slice2(input_dropout_mask, 0), self.W_i)
            xf_t = F.linear(input * get_mask_slice2(input_dropout_mask, 1), self.W_f)
            xc_t = F.linear(input * get_mask_slice2(input_dropout_mask, 2), self.W_c)
            xo_t = F.linear(input * get_mask_slice2(input_dropout_mask, 3), self.W_o)

        if h_dropout_mask is None:
            hi_t = F.linear(h_tm1 * get_mask_slice(self._h_dropout_mask, 0), self.U_i)
            hf_t = F.linear(h_tm1 * get_mask_slice(self._h_dropout_mask, 1), self.U_f)
            hc_t = F.linear(h_tm1 * get_mask_slice(self._h_dropout_mask, 2), self.U_c)
            ho_t = F.linear(h_tm1 * get_mask_slice(self._h_dropout_mask, 3), self.U_o)
        else:
            hi_t = F.linear(h_tm1 * get_mask_slice2(h_dropout_mask, 0), self.U_i)
            hf_t = F.linear(h_tm1 * get_mask_slice2(h_dropout_mask, 1), self.U_f)
            hc_t = F.linear(h_tm1 * get_mask_slice2(h_dropout_mask, 2), self.U_c)
            ho_t = F.linear(h_tm1 * get_mask_slice2(h_dropout_mask, 3), self.U_o)

        if input.is_cuda:
            igates = torch.cat([xi_t, xf_t, xc_t, xo_t], dim=-1)
            hgates = torch.cat([hi_t, hf_t, hc_t, ho_t], dim=-1)
            state = fusedBackend.LSTMFused.apply
            return state(igates, hgates, c_tm1, self.bias_ih, self.bias_hh)
        else:
            i_t = torch.sigmoid(xi_t + self.bias_ih[:self.hidden_size] + hi_t + self.bias_hh[:self.hidden_size])
            f_t = torch.sigmoid(xf_t + self.bias_ih[self.hidden_size:2 * self.hidden_size] + hf_t + self.bias_hh[self.hidden_size:2 * self.hidden_size])
            c_t = f_t * c_tm1 + i_t * torch.tanh(xc_t + self.bias_ih[2 * self.hidden_size:3 * self.hidden_size] + hc_t + self.bias_hh[2 * self.hidden_size:3 * self.hidden_size])
            o_t = torch.sigmoid(xo_t + self.bias_ih[3 * self.hidden_size:4 * self.hidden_size] + ho_t + self.bias_hh[3 * self.hidden_size:4 * self.hidden_size])
            h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
    
    
class ParentFeedingLSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size):
        super(ParentFeedingLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_i = Parameter(torch.Tensor(hidden_size, input_size))
        self.U_i = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U_i_p = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = Parameter(torch.Tensor(hidden_size))

        self.W_f = Parameter(torch.Tensor(hidden_size, input_size))
        self.U_f = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U_f_p = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(torch.Tensor(hidden_size))
        self.b_f_p = Parameter(torch.Tensor(hidden_size))

        self.W_c = Parameter(torch.Tensor(hidden_size, input_size))
        self.U_c = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U_c_p = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = Parameter(torch.Tensor(hidden_size))

        self.W_o = Parameter(torch.Tensor(hidden_size, input_size))
        self.U_o = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U_o_p = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal(self.W_i)
        init.orthogonal(self.U_i)
        init.orthogonal(self.U_i_p)

        init.orthogonal(self.W_f)
        init.orthogonal(self.U_f)
        init.orthogonal(self.U_f_p)

        init.orthogonal(self.W_c)
        init.orthogonal(self.U_c)
        init.orthogonal(self.U_c_p)

        init.orthogonal(self.W_o)
        init.orthogonal(self.U_o)
        init.orthogonal(self.U_o_p)

        self.b_i.data.fill_(0.)
        self.b_c.data.fill_(0.)
        self.b_o.data.fill_(0.)
        # forget bias set to 1.
        self.b_f.data.fill_(1.)
        self.b_f_p.data.fill_(1.)

    def forward(self, input, hidden_states):
        h_tm1, c_tm1, h_tm1_p, c_tm1_p = hidden_states
        i_t = torch.sigmoid(F.linear(input, self.W_i) + F.linear(h_tm1, self.U_i) + F.linear(h_tm1_p, self.U_i_p) + self.b_i)

        xf_t = F.linear(input, self.W_f)
        f_t = torch.sigmoid(xf_t + F.linear(h_tm1, self.U_f) + self.b_f)
        f_t_p = torch.sigmoid(xf_t + F.linear(h_tm1_p, self.U_f_p) + self.b_f_p)

        xc_t = F.linear(input, self.W_c) + F.linear(h_tm1, self.U_c) + F.linear(h_tm1_p, self.U_c_p) + self.b_c
        c_t = f_t * c_tm1 + f_t_p * c_tm1_p + i_t * torch.tanh(xc_t)

        o_t = torch.sigmoid(F.linear(input, self.W_o) + F.linear(h_tm1, self.U_o) + F.linear(h_tm1_p, self.U_o_p) + self.b_o)
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, dropout=0., cell_factory=RecurrentDropoutLSTMCell):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.cell_factory = cell_factory
        num_directions = 2 if bidirectional else 1
        self.lstm_cells = []

        for direction in range(num_directions):
            cell = cell_factory(input_size, hidden_size, dropout=dropout)
            self.lstm_cells.append(cell)

            suffix = '_reverse' if direction == 1 else ''
            cell_name = 'cell{}'.format(suffix)
            self.add_module(cell_name, cell)

    def forward(self, input, hidden_state=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            max_batch_size = input.size(1)

        for cell in self.lstm_cells:
            cell.set_dropout_masks(max_batch_size)

        if hidden_state is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(num_directions,
                                 max_batch_size, self.hidden_size,
                                 requires_grad=False)
            hidden_state = (hx, hx)

        rec_factory = variable_recurrent_factory if is_packed else Recurrent
        if self.bidirectional:
            layer = (rec_factory(self.cell),
                     rec_factory(self.cell_reverse, reverse=True))
        else:
            layer = (rec_factory(self.cell),)

        func = StackedRNN(layer,
                          num_layers=1,
                          lstm=True,
                          dropout=0.,
                          train=self.training)
        next_hidden, output = func(input, hidden_state, weight=[[], []], batch_sizes=batch_sizes)

        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, next_hidden
