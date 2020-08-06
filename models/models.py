import torch
from torch import nn
from utils import load_class

from models import modules
from dataloader.data import TabularFeature, JointTabularFeature


class MultitaskFinetune(nn.Module):
    def __init__(self, joint_vocab, tables, timestep_agg_modelcls='models.TimestepEncoder', patient_modelcls='models.PatientRNNEncoder', **kwargs):
        super().__init__()

        tabular_tables = [table for table in tables if isinstance(table, (TabularFeature, JointTabularFeature))]
        self.timestep_encoder = load_class(timestep_agg_modelcls)(joint_vocab, tables=tabular_tables, **kwargs)
        self.patient_encoder = load_class(patient_modelcls)(self.timestep_encoder.out_size, tables=tabular_tables, **kwargs)
        self.predictor = modules.BenchmarkPredictor(self.patient_encoder.out_size, **kwargs)

    def forward(self, dem, *tables):
        '''
        Input:
            chartevents: Tensor for chart table with shape: N, L, C
            labevents: Tensor for chart table with shape: N, L, C
            outputevents: Tensor for chart table with shape: N, L, C
            dem: Tensor for demographics with shape: N, C
        Output:
            Tuple of (Dictionary of predictions, Embeddings)
        '''
        outputs = {}
        timesteps, output = self.timestep_encoder(*tables)  # N, L, C
        outputs.update(output)

        patient_timesteps, output = self.patient_encoder(timesteps, dem)  # N, L, C
        outputs.update(output)
        preds, _ = self.predictor(patient_timesteps)
        outputs.update({'patient': patient_timesteps.detach(),
                        'timesteps': torch.cat(timesteps, -1).detach()})
        return preds, outputs
