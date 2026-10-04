"""PlateExtract: native RGBA extraction, fine-tuned from Qwen Image 2.1.

Built with Qwen. Adapter/model weights use the Qwen Research License; this
inference code is MIT licensed. Uses the exact four-step training sampler and
per-channel fp8 base weights. Encoder and transformer alternate on one GPU.
"""
import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import time
import sys
import traceback
import types

import numpy as np
from PIL import Image
import torch
from torch import nn
from safetensors.torch import load_file

BASE_REPO = "Qwen/Qwen-Image-2.1"
BASE_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
TURBO_REPO = "Viggle/Qwen-Image-2.1-viggle-turbo"
TURBO_REVISION = "b77064be8b3f0b1a13c6a212067cb3d281c60c84"
HF_REPO = "trmz/plate-extract-qwen-image-2.1"
WEIGHT_NAME = "loras/extract.safetensors"
PROMPT = ("Extract the foreground from Picture 1 using Picture 2 as its clean background plate. "
          "Output the extracted foreground as an RGBA image with a transparent background, "
          "at exactly the same position and size. Preserve its colors, details, and soft alpha edges.")
DT = torch.bfloat16


def lora_path():
    override = os.environ.get("QWEN_EXTRACT_LORA")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"QWEN_EXTRACT_LORA={path} does not exist")
        return str(path)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(os.environ.get("QWEN_HF_REPO", HF_REPO), WEIGHT_NAME,
                           revision=os.environ.get("QWEN_HF_REVISION") or None)


class Fp8Linear(nn.Module):
    """Same weight-only per-output-channel quantization used during training."""
    def __init__(self, linear):
        super().__init__()
        weight = linear.weight.detach().float()
        scale = (weight.abs().amax(dim=1, keepdim=True) / 448).clamp(min=1e-12)
        self.register_buffer("w8", (weight / scale).to(torch.float8_e4m3fn))
        self.register_buffer("scale", scale.to(DT))
        self.bias = linear.bias
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x):
        return nn.functional.linear(x, self.w8.to(x.dtype) * self.scale, self.bias)


def quantize_fp8(module, device, skip=("visual", "lm_head")):
    for name, child in list(module.named_children()):
        if name in skip:
            child.to(device)
        elif isinstance(child, nn.Linear):
            setattr(module, name, Fp8Linear(child.to(device)))
        else:
            quantize_fp8(child, device, skip)
            for param in list(child.parameters(recurse=False)) + list(child.buffers(recurse=False)):
                param.data = param.data.to(device)


class AdapterLinear(nn.Module):
    def __init__(self, base, a, b):
        super().__init__()
        self.base = base
        self.register_buffer("a", a.float())
        # alpha/rank is already folded into B in the exported checkpoint.
        self.register_buffer("b", b.float())

    def forward(self, x):
        return self.base(x) + (x.float() @ self.a.t() @ self.b.t()).to(x.dtype)


def attach_adapter(transformer, filename):
    state = load_file(filename)
    keys = [key for key in state if key.endswith(".lora_A.weight")]
    if not keys or len(state) != 2 * len(keys):
        raise ValueError("Expected paired A/B extraction adapter tensors")
    for key in keys:
        name = key.removeprefix("transformer.").removesuffix(".lora_A.weight")
        parent_name, child_name = name.rsplit(".", 1)
        parent = transformer.get_submodule(parent_name)
        base = getattr(parent, child_name)
        a, b = state[key], state[key.replace("lora_A", "lora_B")]
        if a.shape[1] != base.in_features or b.shape != (base.out_features, a.shape[0]):
            raise ValueError(f"Incompatible adapter shape for {name}")
        setattr(parent, child_name, AdapterLinear(base, a, b))
    return len(keys)


def _patch_visual_encoder(patch_embed):
    # Conv3d kernel equals stride. Matmul avoids slow bf16 Conv3d on RTX 50xx.
    # Read the current parameter each time so CPU/GPU swaps remain valid.
    def forward(self, hidden_states):
        weight = self.proj.weight.reshape(self.proj.out_channels, -1)
        return nn.functional.linear(hidden_states.reshape(-1, weight.shape[1]).to(weight.dtype),
                                    weight, self.proj.bias)
    patch_embed.forward = types.MethodType(forward, patch_embed)


class QwenExtractor:
    def __init__(self, device="cuda:0", adapter=None):
        self.device = torch.device(device)
        self.adapter = adapter
        self.pipe = None
        self.modules = 0
        self.load_seconds = None

    def load(self):
        try:
            from diffusers import (AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler,
                                   QwenImage21Pipeline, QwenImage21Transformer2DModel)
            from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        except ImportError as exc:
            raise RuntimeError("Install demo/requirements-qwen.txt in a separate Qwen environment") from exc
        t0 = time.monotonic()
        filename = self.adapter or lora_path()
        te = Qwen3VLForConditionalGeneration.from_pretrained(
            BASE_REPO, subfolder="text_encoder", revision=BASE_REVISION, torch_dtype=DT)
        quantize_fp8(te, self.device)
        _patch_visual_encoder(te.model.visual.patch_embed)
        te.to("cpu").eval()
        torch.cuda.empty_cache()
        vae = AutoencoderKLQwenImage21.from_pretrained(
            BASE_REPO, subfolder="vae", revision=BASE_REVISION, torch_dtype=DT).eval()
        vae.enable_tiling()
        tr = QwenImage21Transformer2DModel.from_pretrained(
            TURBO_REPO, subfolder="transformer", revision=TURBO_REVISION, torch_dtype=DT)
        for block in tr.transformer_blocks:
            quantize_fp8(block, self.device, skip=())
        self.modules = attach_adapter(tr, filename)
        tr.to(self.device).eval()
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            TURBO_REPO, subfolder="scheduler", revision=TURBO_REVISION)
        self.pipe = QwenImage21Pipeline(
            transformer=tr, vae=vae, text_encoder=te,
            processor=Qwen3VLProcessor.from_pretrained(BASE_REPO, subfolder="processor", revision=BASE_REVISION),
            scheduler=scheduler)
        self.load_seconds = time.monotonic() - t0
        print(f"[PlateExtract] loaded {self.modules} adapter modules in {self.load_seconds:.0f}s", flush=True)
        return self

    def unload(self):
        # Our fp8 modules have no external quantization cache. Release them
        # directly rather than copying GPU weights into RAM before disposal.
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def extract(self, composite, background, resolution=512, width=None, height=None,
                seed=0, steps=4, progress=None):
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions, calculate_shift
        if self.pipe is None:
            self.load()
        pipe, dev = self.pipe, self.device
        if composite.size != background.size:
            raise ValueError("Composite and background plate must have matching dimensions")
        if width is None or height is None:
            width, height, _ = calculate_dimensions(resolution**2, composite.width / composite.height)
        if width % 32 or height % 32:
            raise ValueError("Generation dimensions must be multiples of 32")
        if steps != 4:
            raise ValueError("This release was validated with four inference steps")
        # Conditions were opaque RGBA images during training, encoded by the native VAE.
        images = [pipe.image_processor.resize(im.convert("RGBA"), width=width, height=height)
                  for im in (composite, background)]
        if progress:
            progress(0, desc="Encoding Qwen reference images")
        pipe.transformer.to("cpu")
        torch.cuda.empty_cache()
        try:
            pipe.text_encoder.to(dev)
            prompt, prompt_mask, image_mask = pipe.encode_prompt(prompt=PROMPT, image=images, device=dev)
        finally:
            pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()
        pipe.vae.to(dev)
        try:
            def encode(image):
                x = pipe.image_processor.preprocess(image, width=width, height=height).unsqueeze(2).to(dev, DT)
                z = pipe._encode_vae_image(x, None)
                return pipe._pack_latents(z, 1, 64, z.shape[3], z.shape[4])
            condition = torch.cat([encode(im) for im in images], dim=1)
            target_slots = height // 16 * (width // 16)
            generator = torch.Generator("cpu").manual_seed(int(seed))
            latents = torch.randn((1, target_slots, 64), generator=generator).to(dev, DT)
            image_mask = torch.cat([image_mask, image_mask.new_ones(image_mask.shape[0], target_slots // 4)], 1)
            scheduler = pipe.scheduler
            mu = calculate_shift(target_slots, scheduler.config.base_image_seq_len,
                                 scheduler.config.max_image_seq_len, scheduler.config.base_shift, scheduler.config.max_shift)
            scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
            pipe.transformer.to(dev)
            for i, timestep in enumerate(scheduler.timesteps):
                prediction = pipe.transformer(
                    hidden_states=torch.cat([condition, latents], dim=1),
                    timestep=(timestep / 1000).to(dev, DT).expand(1),
                    encoder_hidden_states=prompt,
                    encoder_hidden_states_mask=None if prompt_mask is None else prompt_mask.to(dev),
                    img_shapes=[[(1, height // 16, width // 16)] * 3], img_mask=image_mask,
                    return_dict=False)[0][:, -target_slots:]
                latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0]
                if progress:
                    progress((i + 1) / steps, desc=f"Qwen extraction step {i + 1}/{steps}")
            z = latents.transpose(1, 2).reshape(1, 64, 1, height // 16, width // 16)
            mean = torch.tensor(pipe.vae.config.latents_mean, device=dev, dtype=DT).view(1, 64, 1, 1, 1)
            std = torch.tensor(pipe.vae.config.latents_std, device=dev, dtype=DT).view(1, 64, 1, 1, 1)
            decoded = pipe.vae.decode(z * std + mean, return_dict=False)[0][:, :, 0]
            return pipe.image_processor.postprocess(decoded, output_type="pil")[0].convert("RGBA")
        finally:
            pipe.vae.to("cpu")
            torch.cuda.empty_cache()


def worker():
    """Separate environment keeps Qwen dependencies independent of ai-toolkit."""
    engine = None
    def send(message):
        print(json.dumps(message), file=sys.__stdout__, flush=True)
    for line in sys.stdin:
        try:
            command = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if command["action"] == "load":
                    engine = QwenExtractor().load()
                    response = {"ok": True, "modules": engine.modules, "load_seconds": engine.load_seconds}
                elif command["action"] == "extract":
                    if engine is None:
                        raise RuntimeError("Load the model before extraction")
                    torch.cuda.reset_peak_memory_stats(engine.device)
                    def progress(value, desc):
                        send({"progress": value, "desc": desc})
                    with Image.open(command["composite"]) as comp, Image.open(command["background"]) as bg:
                        rgba = engine.extract(comp, bg, width=command["width"], height=command["height"],
                                              seed=command.get("seed", 0), progress=progress)
                    rgba.save(command["output"])
                    response = {"ok": True, "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}
                elif command["action"] == "unload":
                    if engine is not None:
                        engine.unload()
                    send({"ok": True})
                    return
                else:
                    raise ValueError("Unknown worker command")
            send(response)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            send({"ok": False, "error": str(exc)})


def main():
    if "--worker" in sys.argv:
        worker()
        return
    ap = argparse.ArgumentParser(description="PlateExtract — native RGBA extraction, built with Qwen")
    ap.add_argument("--composite", required=True)
    ap.add_argument("--background", required=True)
    ap.add_argument("--output", default="extracted.png")
    ap.add_argument("--resolution", type=int, default=512, choices=(512, 768, 1024))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lora", default=None)
    args = ap.parse_args()
    engine = QwenExtractor(args.device, args.lora).load()
    with Image.open(args.composite) as composite, Image.open(args.background) as background:
        result = engine.extract(composite, background, resolution=args.resolution, seed=args.seed)
    result.save(args.output)
    print(f"Saved RGBA PNG to {args.output}")


if __name__ == "__main__":
    main()
