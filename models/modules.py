import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from utils import load_class


def nonzero_avg_pool1d(x, kernel_size, stride=1):
    x = x.unfold(2, kernel_size, stride)
    div = (x.detach() != 0).sum(dim=-1).float()
    x = x.sum(dim=-1) / (div + 1e-5)
    return x


def nonzero_avg_pool(x, inp):
    mask = (inp.detach()[:, :, :100] == 0).all(2)
    x = x * mask[:, None]
    div = (inp.size(1) - mask.sum(dim=1, keepdim=True)).float()
    mask = (div.detach() != 0)
    x = (x.sum(dim=-1) * mask) / (div + 1e-5)
    return x[:, :, None]


def norm_sum_pool(x):
    norm = torch.sqrt(torch.abs(x.sum(dim=-1, keepdim=True)))
    x = x.sum(dim=-1, keepdim=True) / (norm + 1e-5)
    return x


def norm_sum_pool1d(x, kernel_size, stride=1):
    x = x.unfold(2, kernel_size, stride)
    norm = torch.sqrt(torch.abs(x.sum(dim=-1)))
    x = x.sum(dim=-1) / (norm + 1e-5)
    return x


class DeepCNN(nn.Module):
    def __init__(self, input_size, hidden_size, drop_p, kernel_sizes=[[1, 4, 12], [1, 4]]):
        super().__init__()
        self.hidden_size = hidden_size

        convs = []
        in_c = input_size
        for kernel_size in kernel_sizes:
            convs.append(nn.ModuleList([nn.Conv1d(in_channels=in_c,
                                                  out_channels=hidden_size,
                                                  kernel_size=k) for k in kernel_size]))
            in_c = hidden_size*len(kernel_size)

        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, input):
        out = input
        for i, conv in enumerate(self.convs, 1):
            out = [F.relu(conv_(F.pad(out, (conv_.weight.shape[-1]-1, 0)))) for conv_ in conv]

            last_conv = i == len(self.convs)

            # out = [F.max_pool1d(c, 3 if not last_conv else c.size(-1), 2) for c in out]
            out = torch.cat(out, dim=1)

        out = self.dropout(out)
        return out


class BenchmarkPredictor(nn.Module):
    def __init__(self, input_size, decision_dropout, **otherkw):
        super().__init__()
        self.decision_mlps = nn.ModuleDict(dict(in_hospital_mortality=nn.Linear(input_size, 1),
                                                length_of_stay_classification=nn.Conv1d(input_size, 10, 1),
                                                length_of_stay_regression=nn.Conv1d(input_size, 1, 1),
                                                phenotyping=nn.Linear(input_size, 25),
                                                decompensation=nn.Conv1d(input_size, 1, 1)))
        self.decision_dropout = nn.Dropout(decision_dropout)

    def forward(self, input):
        '''
        Input:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        Output:
            Dictionary of predictions
        '''
        input = self.decision_dropout(input)

        last_timestep = input[:, -1]
        if input.shape[1] > 47:
            ihm_timestep = input[:, 47]
        else:
            ihm_timestep = input[:, 0]

        preds = {}
        preds['phenotyping'] = self.decision_mlps['phenotyping'](last_timestep)  # N, 25
        preds['in_hospital_mortality'] = self.decision_mlps['in_hospital_mortality'](ihm_timestep)  # N, 1
        preds['length_of_stay_classification'] = self.decision_mlps['length_of_stay_classification'](input.transpose(1, 2)).transpose(1, 2)  # N, L, 10
        preds['length_of_stay_regression'] = self.decision_mlps['length_of_stay_regression'](input.transpose(1, 2)).squeeze(1)  # N, L
        preds['decompensation'] = self.decision_mlps['decompensation'](input.transpose(1, 2)).squeeze(1)  # N, L

        return preds, {}


class MaxMeanSumPool(nn.Module):
    def __init__(self, input_size, input_dropout, **otherkws):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)
        self.d = input_size
        self.out_size = input_size * 3

    def forward(self, input):
        input = self.input_dropout(input)
        out = F.relu(input).transpose(1, 2)
        max_ind = []
        o_max, mi_ = F.max_pool1d(out, out.size(-1), return_indices=True)
        # o_avg = nonzero_avg_pool(out, input)
        o_avg = F.avg_pool1d(out, out.size(-1))
        o_sum = norm_sum_pool(out)
        max_ind.append(mi_)
        out = torch.cat([o_max, o_avg, o_sum], 1).squeeze(-1)
        max_ind = torch.cat(max_ind, dim=1)
        return out, max_ind


class LinearMaxMeanSumPool(nn.Module):
    def __init__(self, input_size, hidden_size, input_dropout=0.0, padaware=False, **otherkws):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(input_dropout)
        self.linear = nn.Linear(input_size, hidden_size)
        self.d = hidden_size
        self.out_size = hidden_size * 3

        self.padaware = padaware

    def forward(self, input):
        '''
        Input:
            input: T, L, C
        '''
        input = self.input_dropout(input)  # T, L, C
        out = F.relu(self.linear(input))  # T, L, C
        if self.padaware:
            mask = (input[:, :, :100].detach() != 0).all(2).byte()
            out = (out * mask[:, :, None])
        out = out.transpose(1, 2).contiguous()  # T, C, L
        o_max, max_ind = F.max_pool1d(out, out.size(-1), return_indices=True)
        o_max = o_max.squeeze(-1)
        if self.padaware:
            o_avg = nonzero_avg_pool(out, input).squeeze(-1)
        else:
            o_avg = F.avg_pool1d(out, out.size(-1)).squeeze(-1)
        o_sum = norm_sum_pool(out).squeeze(-1)
        outs = [o_max, o_avg, o_sum]

        out = torch.cat(outs, 1)  # T, C
        return out, max_ind


class DeepMaxMeanSumPool(nn.Module):
    def __init__(self, input_size, hidden_size, input_dropout=0.0, padaware=False, **otherkws):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(input_dropout)
        self.linear = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size))
        self.d = hidden_size
        self.out_size = hidden_size * 3

        self.padaware = padaware

    def forward(self, input):
        input = self.input_dropout(input)
        out = F.relu(self.linear(input))
        if self.padaware:
            mask = (input[:, :,:100].detach() != 0).all(2).byte()
            out = (out * mask[:, :, None])
        out = out.transpose(1, 2)
        o_max, max_ind = F.max_pool1d(out, out.size(-1), return_indices=True)
        o_max = o_max.squeeze(-1)
        if self.padaware:
            o_avg = nonzero_avg_pool(out, input).squeeze(-1)
        else:
            o_avg = F.avg_pool1d(out, out.size(-1)).squeeze(-1)
        o_sum = norm_sum_pool(out).squeeze(-1)
        out = torch.cat([o_max, o_avg, o_sum], 1)
        return out, max_ind


class LinearMaxSumPool(nn.Module):
    def __init__(self, input_size, hidden_size, input_dropout, **otherkws):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(input_dropout)
        self.linear = nn.Linear(input_size, hidden_size)
        self.d = hidden_size
        self.out_size = hidden_size * 2

    def forward(self, input):
        input = self.input_dropout(input)
        out = F.relu(self.linear(input)).transpose(1, 2)
        max_ind = []
        o_max, mi_ = F.max_pool1d(out, out.size(-1), return_indices=True)
        # o_avg = nonzero_avg_pool(out, input)
        o_sum = norm_sum_pool(out)
        max_ind.append(mi_)
        out = torch.cat([o_max, o_sum], 1).squeeze(-1)
        max_ind = torch.cat(max_ind, dim=1)
        return out, max_ind


class LinearMaxPadMeanSumPool(nn.Module):
    def __init__(self, input_size, hidden_size, input_dropout, **otherkws):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(input_dropout)
        self.linear = nn.Linear(input_size, hidden_size)
        self.d = hidden_size
        self.out_size = hidden_size * 3

    def forward(self, input):
        input = self.input_dropout(input)
        out = F.relu(self.linear(input)).transpose(1, 2)
        max_ind = []
        o_max, mi_ = F.max_pool1d(out, out.size(-1), return_indices=True)
        o_avg = nonzero_avg_pool(out, input)
        o_sum = norm_sum_pool(out)
        max_ind.append(mi_)
        out = torch.cat([o_max, o_avg, o_sum], 1).squeeze(-1)
        max_ind = torch.cat(max_ind, dim=1)
        return out, max_ind


class LinearMaxPool(nn.Module):
    def __init__(self, input_size, hidden_size, input_dropout, **otherkws):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(input_dropout)
        self.linear = nn.Linear(input_size, hidden_size)
        self.out_size = hidden_size

    def forward(self, input):
        input = self.input_dropout(input)
        out = F.relu(self.linear(input)).transpose(1, 2)
        out, max_ind = F.max_pool1d(out, out.size(-1), return_indices=True)
        return out.squeeze(-1), max_ind
