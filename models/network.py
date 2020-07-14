import sys

import torch
from torch import nn
from torch.nn import functional as F

from models.backbone import CNNEncoder
from utils.utils import init_weights

sys.dont_write_bytecode = True


class AEAModule(nn.Module):
    def __init__(self, inplanes, scale_value=50, from_value=0.4, value_interval=0.5):
        super(AEAModule, self).__init__()
        self.inplanes = inplanes
        self.scale_value = scale_value
        self.from_value = from_value
        self.value_interval = value_interval

        self.f_psi = nn.Sequential(
            nn.Linear(self.inplanes, self.inplanes // 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.inplanes // 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, f_x):
        # the Eq.(7) should be an approximation of Step Function with the adaptive threshold,
        # please refer to https://github.com/LegenDong/ATL-Net/pdf/ATL-Net_Update.pdf
        b, hw, c = x.size()
        clamp_value = self.f_psi(x.view(b * hw, c)) * self.value_interval + self.from_value
        clamp_value = clamp_value.view(b, hw, 1)
        clamp_fx = torch.sigmoid(self.scale_value * (f_x - clamp_value))
        attention_mask = F.normalize(clamp_fx, p=1, dim=-1)

        return attention_mask


class ATLModule(nn.Module):
    def __init__(self, inplanes, transfer_name='W', scale_value=30, atten_scale_value=50, from_value=0.5,
                 value_interval=0.3):
        super(ATLModule, self).__init__()

        self.inplanes = inplanes
        self.scale_value = scale_value

        if transfer_name == 'W':
            self.W = nn.Sequential(
                nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.inplanes),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            raise RuntimeError

        self.attention_layer = AEAModule(self.inplanes, atten_scale_value, from_value, value_interval)

    def forward(self, query_data, support_data):
        b, c, h, w = query_data.size()
        s, _, _, _ = support_data.size()
        support_data = support_data.unsqueeze(0).expand(b, -1, -1, -1, -1).contiguous().view(b * s, c, h, w)

        w_query = self.W(query_data).view(b, c, h * w)
        w_query = w_query.permute(0, 2, 1).contiguous()
        w_support = self.W(support_data).view(b, s, c, h * w).permute(0, 2, 1, 3).contiguous().view(b, c, s * h * w)
        w_query = F.normalize(w_query, dim=2)
        w_support = F.normalize(w_support, dim=1)

        f_x = torch.matmul(w_query, w_support)
        attention_score = self.attention_layer(w_query, f_x)

        query_data = query_data.view(b, c, h * w).permute(0, 2, 1)
        support_data = support_data.view(b, s, c, h * w).permute(0, 2, 1, 3).contiguous().view(b, c, s * h * w)
        query_data = F.normalize(query_data, dim=2)
        support_data = F.normalize(support_data, dim=1)

        match_score = torch.matmul(query_data, support_data)
        attention_match_score = torch.mul(attention_score, match_score).view(b, h * w, s, h * w).permute(0, 2, 1, 3)

        final_local_score = torch.sum(attention_match_score.contiguous().view(b, s, h * w, h * w), dim=-1)
        final_score = torch.mean(final_local_score, dim=-1) * self.scale_value

        return final_score, final_local_score


class ALTNet(nn.Module):
    def __init__(self, base_model='Conv64F', base_model_info=None, **kwargs):
        super(ALTNet, self).__init__()

        if base_model_info is None:
            base_model_info = {}
        self.base_model = base_model
        self.base_model_info = base_model_info
        self.kwargs = kwargs
        self._init_module()

    def _init_module(self):
        if self.base_model == 'Conv64F':
            self.features = CNNEncoder(**self.base_model_info)
        else:
            raise RuntimeError

        self.metric_layer = ATLModule(**self.kwargs)

        init_weights(self, init_type='normal')

    def forward(self, query_data, support_data):
        query_feature = self.features(query_data)
        support_feature = []
        for support in support_data:
            support_feature.append(self.features(support))

        scores = []
        local_scores = []
        for support in support_feature:
            score, local_score = self.metric_layer(query_feature, support)
            scores.append(score)
            local_scores.append(local_score)

        scores = torch.cat(scores, 1)
        local_scores = torch.cat(local_scores, 1)

        return query_feature, scores, local_scores


if __name__ == '__main__':
    model = ALTNet(base_model='Conv64F', inplanes=64)
    query_data = torch.rand(75, 3, 84, 84)
    support_data = [torch.rand(25, 3, 84, 84)]
    result = model(query_data, support_data)
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('  + Number of params: %.3fM' % (trainable_num / 1e6))
    print('  + Number of params: {}'.format(trainable_num))
