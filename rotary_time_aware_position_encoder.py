import torch.nn as nn
import torch
import math


class RoTAPE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 4 == 0
        # [d/4]
        self.theta = torch.exp(torch.arange(0, d_model, 4, device='cuda:0') * -(math.log(10000.0) / d_model))

    def get_position(self, t_d, t_h):
        # [b, n, d/4]
        pos_day = t_d.unsqueeze(-1) * self.theta
        pos_hour = t_h.unsqueeze(-1) * self.theta
        cos_day = torch.cos(pos_day).unsqueeze(1)
        sin_day = torch.sin(pos_day).unsqueeze(1)
        cos_hour = torch.cos(pos_hour).unsqueeze(1)
        sin_hour = torch.sin(pos_hour).unsqueeze(1)
        return cos_day, sin_day, cos_hour, sin_hour

    def forward(self, x, t_d, t_h):
        cos_day, sin_day, cos_hour, sin_hour = self.get_position(t_d, t_h)
        # [b, 1, n, d/4]
        x_1, x_2, x_3, x_4 = x[..., 0::4], x[..., 1::4], x[..., 2::4], x[..., 3::4]
        x_day_1 = x_1 * cos_day + x_2 * sin_day
        x_day_2 = x_2 * cos_day - x_1 * sin_day
        x_hour_3 = x_3 * cos_hour + x_4 * sin_hour
        x_hour_4 = x_4 * cos_hour - x_3 * sin_hour
        x = torch.cat([x_day_1, x_day_2, x_hour_3, x_hour_4], dim=-1)
        return x