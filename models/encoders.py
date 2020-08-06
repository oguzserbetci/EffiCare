import torch
import torch.nn

from models.modules import *
from utils import load_class


class EventEncoder(nn.Module):
    def __init__(self, vocab, freeze_emb, emb_dim, include_time, n_bins):
        super().__init__()
        input_size = emb_dim
        input_size += n_bins
        if include_time:
            input_size += 2

        self.vocab = vocab
        if vocab.vectors is not None:
            self.encoder = nn.Embedding.from_pretrained(vocab.vectors, freeze=freeze_emb, padding_idx=0)
        else:
            self.encoder = nn.Embedding(len(vocab), emb_dim, padding_idx=0)
            if freeze_emb:
                self.encoder.weight.requires_grad = False

        bins = torch.eye(n_bins)
        bins = torch.cat([torch.zeros(1, n_bins), bins], 0)
        self.bins = nn.Parameter(bins, requires_grad=False)

        self.include_time = include_time

    def forward(self, input):
        if input.shape[-1] == 100:
            emb = self.encoder(input)
        else:
            emb = self.encoder(input[:, :, 0])
        arr = [emb]
        if self.include_time:
            time_input = torch.arange(len(input)).float()[:, None, None].expand(input.shape[:2] + (1,)).to(next(self.parameters()).device)
            arr.extend([torch.log(time_input + 1), torch.exp(time_input/1000) - 1])
        sep_bins = self.bins[input[:, :, 1].long()]
        arr.append(sep_bins)
        emb = torch.cat(arr, -1)  # N, L, C
        return emb


class TimestepEncoder(nn.Module):
    def __init__(self, joint_vocab, tables, freeze_emb=True, emb_dim=100, include_time=True, n_bins=9, hidden_size=50,
                 timestep_modelcls='models.LinearMaxMeanSumPool', freeze_encoders=False,
                 **kwargs):
        super().__init__()

        input_size = emb_dim
        if include_time:
            input_size += 2
        input_size += n_bins

        self.event_encoder = EventEncoder(joint_vocab, freeze_emb, emb_dim, include_time, n_bins)

        models = []

        self.out_size = 0
        for _ in tables:
            model = load_class(timestep_modelcls)(input_size, hidden_size=hidden_size, **kwargs)
            models.append(model)
            self.out_size += model.out_size
        self.models = nn.ModuleList(models)

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.tables = tables

    def forward(self, *inputs: Tensor):
        '''
        Input:
            inputs: Tensors for table with shape: N, L, C
        Output:
            Tensor of timestep embeddings with shape: N, L, C
        '''
        out = []
        measure_inds = {}
        for i, (table, input, model) in enumerate(zip(self.tables, inputs, self.models)):
            batch_size = input.size(0)
            input = input.reshape((-1, ) + input.shape[2:])  # NxL1, L2, C
            input = self.event_encoder(input)
            input, measure_ind = model(input)  # NxL, C
            measure_inds[table.table] = measure_ind

            input = input.reshape((batch_size, -1) + input.shape[1:])  # N, L, C
            out.append(input)

        return out, {'measure_inds': measure_inds}


class DeepTimestepEncoder(nn.Module):
    def __init__(self, joint_vocab, tables, freeze_emb=True, emb_dim=100, include_time=True, n_bins=9, hidden_size=50,
                 timestep_modelcls='models.LinearMaxMeanSumPool', freeze_encoders=False,
                 pat_hidden_size=128,
                 **kwargs):
        super().__init__()

        input_size = emb_dim
        input_size += n_bins
        if include_time:
            input_size += 2

        self.event_encoder = EventEncoder(joint_vocab, freeze_emb, emb_dim, include_time, n_bins)
        self.model = load_class(timestep_modelcls)(input_size, hidden_size=hidden_size, **kwargs)
        self.mixer = nn.Sequential(nn.Linear(len(tables) * self.model.out_size, self.model.out_size),
                                   nn.ReLU(),
                                   nn.Linear(self.model.out_size, pat_hidden_size),
                                   nn.ReLU())

        for param in self.model.parameters():
            param.requires_grad = False

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.out_size = pat_hidden_size
        self.tables = tables

    def forward(self, *inputs: Tensor):
        '''
        Input:
            input: Tensors for table with shape: N, L, C
        Output:
            Tensor of timestep embeddings with shape: N, L, C
        '''
        out = []
        measure_inds = {}
        for table, input in zip(self.tables, inputs):
            batch_size = input.size(0)
            input = input.reshape((-1, ) + input.shape[2:])  # NxL1, L2, C
            input = self.event_encoder(input)
            input, measure_ind = self.model(input)  # NxL, C
            measure_inds[table.table] = measure_ind

            input = input.reshape((batch_size, -1) + input.shape[1:])  # N, L, C
            out.append(input)

        timesteps = torch.cat(out, -1)
        timesteps = self.mixer(timesteps)
        return timesteps, {'measure_inds': measure_inds}


class PatientPoolEncoder(nn.Module):
    def __init__(self, input_size, tables, pat_padaware=False, timestep_modelcls='models.LinearMaxMeanSumPool', include_demfc=True, dem_size=8, dem_dropout=.0, visit_dropout=.0, freeze_encoders=False, **otherkw):
        super().__init__()

        if include_demfc:
            self.dem_fc = nn.Sequential(nn.Linear(dem_size, 40),
                                        nn.ReLU(),
                                        nn.Dropout(dem_dropout),
                                        nn.Linear(40, 20),
                                        nn.ReLU()
                                        )
            dem_size = 20

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.include_demfc = include_demfc
        self.out_size = 3 * input_size + dem_size
        self.padaware = pat_padaware
        self.tables = tables

    def forward(self, input, dem):
        '''
        Input:
            input: Tensor of timestep embeddings with shape: N, L, C
            dem: Tensor of demographics with shape: N, C
        Output:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        '''
        patient = []
        time_inds = {}
        activations = {}
        for inp, table in zip(input, self.tables):
            N = inp.shape[0]
            L = inp.shape[1]

            inp = F.pad(inp.contiguous(), (0, 0, L-1, 0, 0, 0))  # N, L', C
            inp = inp.transpose(1, 2).contiguous()  # N, C, L

            p_max, p_max_ind = F.max_pool1d(inp, L, 1, return_indices=True)  # N, C, L
            p_max_ind = (p_max_ind - (L - 1)).detach()

            # Collect max activations and indices
            time_inds[table.table] = p_max_ind
            activations[table.table] = p_max.detach()

            p_max = p_max.transpose(1, 2).contiguous()  # N, L, C
            if self.padaware:
                p_avg = nonzero_avg_pool1d(inp, L, 1).transpose(1, 2).contiguous()  # N, C, L
            else:
                p_avg = F.avg_pool1d(inp, L, 1).transpose(1, 2).contiguous()  # N, L, C
            p_sum = norm_sum_pool1d(inp, L, 1).transpose(1, 2).contiguous()  # N, L, C

            patient_timesteps = torch.cat([p_max, p_avg, p_sum], -1)  # N, L, C'
            patient.append(patient_timesteps)

        if self.include_demfc:
            dem = self.dem_fc(dem)
        dem = dem.unsqueeze(1).expand(N, L, -1)  # N, L, C
        patient_timesteps = torch.cat(patient + [dem], 2)  # N, L, C'

        return patient_timesteps, {'time_inds': time_inds,
                                   'activations': activations}


class PatientRNNEncoder(nn.Module):
    def __init__(self, input_size, include_dem=True, include_demfc=True, dem_size=10, dem_dropout=.0, pat_hidden_size=128, visit_dropout=.0, pat_layers=1, freeze_encoders=False, **otherkw):
        super().__init__()
        if include_dem:
            if include_demfc:
                self.dem_fc = nn.Sequential(nn.Linear(dem_size, 40),
                                            nn.ReLU(),
                                            nn.Dropout(dem_dropout),
                                            nn.Linear(40, 20),
                                            nn.ReLU()
                                            )
                dem_size = 20
            input_size += dem_size
        self.rnn = nn.GRU(input_size, pat_hidden_size, pat_layers, dropout=visit_dropout, batch_first=True)

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.include_dem = include_dem
        self.include_demfc = include_demfc
        self.out_size = pat_hidden_size

    def forward(self, timesteps, dem=None):
        '''
        Input:
            input: Tensor of timestep embeddings with shape: N, L, C
            dem: Tensor of demographics with shape: N, C
        Output:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        '''

        if self.include_dem:
            N = timesteps.shape[0]
            L = timesteps.shape[1]
            if self.include_demfc:
                dem = self.dem_fc(dem)
            dem = dem.unsqueeze(1).expand(N, L, -1)  # N, L, C

            timesteps = torch.cat([timesteps, dem], -1)  # N, L, C'

        timesteps, _ = self.rnn(timesteps)

        # L = timestep_dem.shape[1]

        # input = F.pad(timestep_dem.contiguous(), (0, 0, L-1, 0))  # N, L', C
        # input = input.transpose(1, 2).contiguous()  # N, C, L
        # p_avg = nonzero_avg_pool1d(input, L, 1)
        # p_avg = p_avg.transpose(1, 2).contiguous()  # N, L, C

        # out = torch.cat([timestep_dem, p_avg], dim=2)

        # return F.relu(p_avg), {}
        return timesteps, {}


class PatientMeanEncoder(nn.Module):
    def __init__(self, input_size, include_demfc=True, dem_size=10, dem_dropout=.0, pat_hidden_size=128, visit_dropout=.0, pat_layers=1, freeze_encoders=False, **otherkw):
        super().__init__()
        if include_demfc:
            self.dem_fc = nn.Sequential(nn.Linear(dem_size, 40),
                                        nn.ReLU(),
                                        nn.Dropout(dem_dropout),
                                        nn.Linear(40, 20),
                                        nn.ReLU()
                                        )
            dem_size = 20

        input_size += dem_size

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.include_demfc = include_demfc
        self.out_size = input_size

    def forward(self, timesteps, dem):
        '''
        Input:
            input: Tensor of timestep embeddings with shape: N, L, C
            dem: Tensor of demographics with shape: N, C
        Output:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        '''
        N = timesteps.shape[0]
        L = timesteps.shape[1]
        if self.include_demfc:
            dem = self.dem_fc(dem)
        dem = dem.unsqueeze(1).expand(N, L, -1)  # N, L, C

        timestep_dem = torch.cat([timesteps, dem], -1)  # N, L, C'

        L = timestep_dem.shape[1]

        input = F.pad(timestep_dem.contiguous(), (0, 0, L-1, 0))  # N, L', C
        input = input.transpose(1, 2).contiguous()  # N, C, L
        p_avg = nonzero_avg_pool1d(input, L, 1)
        p_avg = p_avg.transpose(1, 2).contiguous()  # N, L, C

        # out = torch.cat([timestep_dem, p_avg], dim=2)

        return F.relu(p_avg), {}


class PatientCNNEncoder(nn.Module):
    def __init__(self, input_size, include_demfc=True, dem_size=10, dem_dropout=.0, pat_hidden_size=128, visit_dropout=.0, pat_layers=1, freeze_encoders=False, **otherkw):
        super().__init__()
        if include_demfc:
            self.dem_fc = nn.Sequential(nn.Linear(dem_size, 40),
                                        nn.ReLU(),
                                        nn.Dropout(dem_dropout),
                                        nn.Linear(40, 20),
                                        nn.ReLU()
                                        )
            dem_size = 20

        input_size += dem_size
        self.cnn = DeepCNN(input_size, pat_hidden_size, visit_dropout)

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.include_demfc = include_demfc
        self.out_size = pat_hidden_size * 2

    def forward(self, timesteps, dem):
        '''
        Input:
            input: Tensor of timestep embeddings with shape: N, L, C
            dem: Tensor of demographics with shape: N, C
        Output:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        '''
        N = timesteps.shape[0]
        L = timesteps.shape[1]
        if self.include_demfc:
            dem = self.dem_fc(dem)
        dem = dem.unsqueeze(1).expand(N, L, -1)  # N, L, C

        timestep_dem = torch.cat([timesteps, dem], -1)  # N, L, C'

        timestep_dem = self.cnn(timestep_dem.transpose(1,2).contiguous()).transpose(1,2).contiguous()

        # L = timestep_dem.shape[1]

        # input = F.pad(timestep_dem.contiguous(), (0, 0, L-1, 0, 0, 0))  # N, L', C
        # input = input.transpose(1, 2).contiguous()  # N, C, L
        # p_avg = nonzero_avg_pool1d(input, L, 1)
        # p_avg = p_avg.transpose(1, 2).contiguous()  # N, L, C

        # out = torch.cat([timestep_dem, p_avg], dim=2)

        return F.relu(timestep_dem), {}


class PatientRETAINEncoder(nn.Module):
    def __init__(self, input_size, include_dem=True, include_demfc=True, dem_size=10, dem_dropout=.0, pat_hidden_size=128, visit_dropout=.0, pat_layers=1, freeze_encoders=False, **otherkw):
        super().__init__()
        if include_dem:
            if include_demfc:
                self.dem_fc = nn.Sequential(nn.Linear(dem_size, 40),
                                            nn.ReLU(),
                                            nn.Dropout(dem_dropout),
                                            nn.Linear(40, 20),
                                            nn.ReLU()
                                            )
                dem_size = 20
            input_size += dem_size

        # Visit-level attention
        self.a_rnn = nn.GRU(input_size, pat_hidden_size, pat_layers, dropout=visit_dropout, batch_first=True)
        self.a_fc = nn.Linear(pat_hidden_size, 1)

        # Variable-level attention
        self.b_rnn = nn.GRU(input_size, pat_hidden_size, pat_layers, dropout=visit_dropout, batch_first=True)
        self.b_fc = nn.Linear(pat_hidden_size, input_size)

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.include_demfc = include_demfc
        self.include_dem = include_dem
        self.out_size = input_size

    def forward(self, timesteps, dem):
        '''
        Input:
            input: Tensor of timestep embeddings with shape: N, L, C
            dem: Tensor of demographics with shape: N, C
        Output:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        '''
        if self.include_dem:
            N = timesteps.shape[0]
            L = timesteps.shape[1]
            if self.include_demfc:
                dem = self.dem_fc(dem)
            dem = dem.unsqueeze(1).expand(N, L, -1)  # N, L, C

            timesteps = torch.cat([timesteps, dem], -1)  # N, L, C'

        alphas, _ = self.a_rnn(timesteps)
        alphas = self.a_fc(alphas)
        alphas = F.tanh(alphas)

        betas, _ = self.b_rnn(timesteps)
        betas = self.b_fc(betas)
        betas = F.tanh(betas)

        timesteps = timesteps * alphas * betas

        L = timesteps.shape[1]
        input = F.pad(timesteps.contiguous(), (0, 0, L-1, 0))  # N, L', C
        input = input.transpose(1, 2).contiguous()  # N, C, L
        out = norm_sum_pool1d(input, L, 1)
        out = out.transpose(1, 2).contiguous()  # N, L, C

        return out, {}
