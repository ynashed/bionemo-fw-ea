import torch

from bionemo.model.core.infer import BaseEncoderDecoderInference


class NavWrapper(torch.nn.Module):
    def __init__(self, inferer: BaseEncoderDecoderInference):
        super().__init__()
        self.model = inferer.model.cuda()
        self.prepare_for_export()

    def prepare_for_export(self):
        self.model._prepare_for_export()

    def forward(self, tokens_enc, enc_mask):
        return self.model.encode(tokens_enc=tokens_enc, enc_mask=enc_mask)
