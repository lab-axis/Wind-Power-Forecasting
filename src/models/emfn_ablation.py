"""Experimental controls; none replaces the main EMFN or its checkpoints."""
import torch
from torch import nn
from torch.nn import functional as F
from src.models.emfn import EMFN


class SharedAdapterEMFN(EMFN):
    """Tie the two task adapters, preserving both output routes and auxiliaries.

    This tests task-adapter parameter sharing, not removal of every branch.
    Parameter count changes; that is reported rather than called budget-matched.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.regression_mode:
            raise ValueError('Shared-adapter control requires the hurdle head')
        self.zero_adapter = self.mag_adapter


class CommonRepresentationEMFN(EMFN):
    """Same representation architecture/initialization for three output families.

    All heads see weather-state, HF and AR features. End-to-end training lets
    representations diverge; this is NOT a frozen-identical-h experiment.
    """
    def __init__(self, head_kind='hurdle', **kwargs):
        if head_kind not in ('linear','sigmoid','hurdle'):
            raise ValueError('Unknown head family')
        expected_regression = head_kind != 'hurdle'
        if kwargs.get('regression_mode', expected_regression) != expected_regression:
            raise ValueError('Head and loss family disagree')
        kwargs['regression_mode'] = False
        super().__init__(**kwargs)
        self.head_kind = head_kind
        self.regression_mode = expected_regression
        # Remove unused original task adapters and heads from optimization.
        for name in ('mag_adapter','zero_adapter','head_alpha','head_beta','head_zero','head_reg'):
            delattr(self,name)
        width = 2*self.d_model + (4+self.pred_len if self.use_hf_skips else 0) + self.gate_dim
        self.common_adapter = nn.Sequential(nn.Linear(width,self.d_model),nn.GELU())
        out = self.pred_len*(3 if head_kind=='hurdle' else 1)
        self.output_head = nn.Sequential(nn.Linear(self.d_model,64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,out))

    def representation(self,x_target,x_weather):
        x = x_target/self.capacity_mwh
        inputs = torch.cat([x,x_weather],-1) if self.n_weather_features else x
        seq = self.tcn(inputs.transpose(1,2))
        pieces = [seq[:,:,-1],seq.mean(-1)]
        w = min(6,x.shape[1])
        if self.use_hf_skips:
            last = x[:,-1,:]
            lag1 = x[:,-2,:] if x.shape[1]>=2 else last
            lag2 = x[:,-3,:] if x.shape[1]>=3 else lag1
            pieces += [last,last-lag1,last-lag2,x[:,-w:,:].mean(1),self.ar_shortcut(x.squeeze(-1))]
        if self.selective_gate is not None:
            ws = x_weather[:,:,0:1]
            zero = x_target.eq(0).float()
            features = torch.cat([ws[:,-1,:],ws[:,-w:,:].mean(1),ws[:,-w:,:].max(1).values,
                x_weather[:,-1,5:6],x_weather[:,-1,3:4],zero[:,-1,:],zero[:,-w:,:].mean(1),zero.mean(1)],-1)
            pieces.append(self.selective_gate(features))
        return self.common_adapter(torch.cat(pieces,-1))

    def forward(self,x_target,x_weather=None):
        out = self.output_head(self.representation(x_target,x_weather))
        if self.head_kind=='hurdle':
            return out.split(self.pred_len,dim=-1)
        point = torch.sigmoid(out) if self.head_kind=='sigmoid' else out
        return point*self.capacity_mwh,None,None
