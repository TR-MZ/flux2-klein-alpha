import torch
from torch import Tensor, nn


class DiffusersVAEAdapter(nn.Module):
    """Wrap diffusers AutoencoderKLFlux2 with the toolkit VAE interface."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.num_output_channels = vae.config.out_channels
        self.bn_eps = vae.config.batch_norm_eps

    @property
    def device(self):
        return self.vae.device

    @property
    def dtype(self):
        return self.vae.dtype

    @staticmethod
    def _patchify(z: Tensor) -> Tensor:
        b, c, h, w = z.shape
        z = z.view(b, c, h // 2, 2, w // 2, 2)
        z = z.permute(0, 1, 3, 5, 2, 4)
        z = z.reshape(b, c * 4, h // 2, w // 2)
        return z

    @staticmethod
    def _unpatchify(z: Tensor) -> Tensor:
        b, c, h, w = z.shape
        z = z.reshape(b, c // 4, 2, 2, h, w)
        z = z.permute(0, 1, 4, 2, 5, 3)
        z = z.reshape(b, c // 4, h * 2, w * 2)
        return z

    def _normalize(self, z: Tensor) -> Tensor:
        self.vae.bn.eval()
        return self.vae.bn(z)

    def _inv_normalize(self, z: Tensor) -> Tensor:
        self.vae.bn.eval()
        scale = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        mean = self.vae.bn.running_mean.view(1, -1, 1, 1)
        return z * scale.to(z.device, z.dtype) + mean.to(z.device, z.dtype)

    def encode(self, x: Tensor) -> Tensor:
        z = self.vae.encode(x).latent_dist.sample()
        z = self._patchify(z)
        z = self._normalize(z)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = self._inv_normalize(z)
        z = self._unpatchify(z)
        return self.vae.decode(z).sample

    def to(self, *args, **kwargs):
        self.vae.to(*args, **kwargs)
        return self
