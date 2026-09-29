"""Layer-axis sequence packing for Flux.2.

Flux.2's RoPE is already 4-axis -- ``(t, h, w, l)`` -- and the ``t`` axis is what
the base model uses to separate reference images from the image being generated:
the target sits at ``t = 0`` and reference images are pushed out to ``t = 10, 20,
30``. That axis is exactly the layer dimension a layered generator needs, so no
architecture change is required.

The convention here:

    layer i of the generated stack  ->  t = i          (i in 0 .. N-1)
    the conditioning composite      ->  t = COND_T     (10)

With ``N = 1`` this is bit-identical to ordinary Flux.2 image-conditioned
generation, so a LoRA starts from the pretrained behaviour and only has to learn
what ``t = 1 .. N-1`` mean. Everything stays a single joint forward pass: every
layer attends to every other layer and to the conditioning image, which is what
keeps inter-layer consistency without recursion.
"""

import math

import torch
from einops import rearrange
from torch import Tensor

# Time index reserved for the conditioning composite. Matches the base model's
# first reference-image slot, so conditioning behaves as the model already
# expects.
COND_T = 10


def layer_ids(
    num_layers: int,
    h: int,
    w: int,
    device: torch.device,
    t_start: int = 0,
    t_stride: int = 1,
) -> Tensor:
    """Position ids for a stack of layers, shape ``(num_layers * h * w, 4)``.

    Row-major over ``(t, h, w)`` so a plain reshape recovers the layer stack --
    no scatter needed on the way back out.
    """
    t = torch.arange(num_layers, device=device) * t_stride + t_start
    return torch.cartesian_prod(
        t,
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        torch.arange(1, device=device),
    )


def pack_layers(latents: Tensor, t_start: int = 0, t_stride: int = 1) -> tuple[Tensor, Tensor]:
    """``(B, N, C, H, W)`` latents -> tokens ``(B, N*H*W, C)`` and ids ``(B, N*H*W, 4)``."""
    b, n, c, h, w = latents.shape
    tokens = rearrange(latents, "b n c h w -> b (n h w) c")
    ids = layer_ids(n, h, w, latents.device, t_start, t_stride)
    return tokens, ids.unsqueeze(0).expand(b, -1, -1)


def unpack_layers(tokens: Tensor, num_layers: int, h: int, w: int) -> Tensor:
    """Inverse of :func:`pack_layers`."""
    return rearrange(tokens, "b (n h w) c -> b n c h w", n=num_layers, h=h, w=w)


def pack_condition(latent: Tensor, t: int = COND_T) -> tuple[Tensor, Tensor]:
    """``(B, C, H, W)`` conditioning latent -> tokens at a fixed time index."""
    return pack_layers(latent.unsqueeze(1), t_start=t)


def text_ids(seq_len: int, device: torch.device) -> Tensor:
    """Flux.2 puts text tokens on the ``l`` axis with ``t = h = w = 0``."""
    return torch.cartesian_prod(
        torch.arange(1, device=device),
        torch.arange(1, device=device),
        torch.arange(1, device=device),
        torch.arange(seq_len, device=device),
    )


def timestep_shift(t: Tensor, image_seq_len: int, base_shift: float = 0.5, max_shift: float = 1.15) -> Tensor:
    """Resolution-dependent schedule shift, matching ``sampling.get_schedule``.

    ``image_seq_len`` should be the *per-layer* token count: the shift exists to
    compensate for image resolution, and every layer here is the same resolution
    as an ordinary single-image generation.
    """
    m = (max_shift - base_shift) / (4096 - 256)
    mu = m * image_seq_len + base_shift - m * 256
    e = math.exp(mu)
    return e / (e + (1.0 / t.clamp(1e-6, 1.0 - 1e-6) - 1.0))


def sample_timesteps(
    batch_size: int,
    image_seq_len: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
) -> Tensor:
    """Logit-normal timestep sampling, then the resolution shift. Returns 0..1.

    One timestep per *sample*, shared by every layer in that sample -- the stack
    is denoised jointly, so a per-layer noise level would be incoherent.
    """
    u = torch.randn(batch_size, device=device, generator=generator)
    t = torch.sigmoid(u * logit_std + logit_mean)
    return timestep_shift(t, image_seq_len)


def flow_match_inputs(latents: Tensor, timesteps: Tensor, noise: Tensor) -> tuple[Tensor, Tensor]:
    """Rectified-flow interpolation and its velocity target.

    ``latents`` is ``(B, N, C, H, W)``; ``timesteps`` is ``(B,)`` in ``[0, 1]``
    where 1 is pure noise. The target velocity is ``noise - clean``.
    """
    t = timesteps.view(-1, *([1] * (latents.ndim - 1))).to(latents.dtype)
    noisy = (1.0 - t) * latents + t * noise
    return noisy, noise - latents
