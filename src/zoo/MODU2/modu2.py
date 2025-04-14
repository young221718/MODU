import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from src.core import register


__all__ = [
    "MODU2",
]


@register
class MODU2(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder,
        decoder,
        multi_scale=None,
        query_save_path=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.multi_scale = multi_scale
        self.query_save_path = query_save_path
        self.i = 0

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        x = self.backbone(x)
        encoder_out = self.encoder(x)
        output = self.decoder(encoder_out, targets)

        # if self.query_save_path is not None:
        #     query = output["pred_queries"].detach().cpu()
        #     logit = output["pred_logits"].detach().cpu()

        #     save_path = os.path.join(self.query_save_path, f"step_{self.i}.pt")
        #     self.i += 1
        #     torch.save({"queries": query, "logits": logit}, save_path)

        return output

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
