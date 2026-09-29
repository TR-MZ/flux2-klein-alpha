# ai-toolkit-RGBA

My changes to [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) (MIT License, © 2024 Ostris, LLC) for training FLUX.2 Klein adapters with a four-channel (RGBA) VAE. The demo in this repository also uses them for inference.

- `ai-toolkit-rgba.patch`: all changes to existing upstream files, against upstream commit `e03c6e4` ("Fix potential inconsistency with different attention mentods in hidream01"). It includes my dual-GPU training changes and the RGBA changes.
- `new_files/`: files the patch does not contain, because they are new:
  - the RGBA VAE adapter (`extensions_built_in/diffusion_models/flux2/src/vae_adapter.py`) and a helper for layered latents (`layered.py`);
  - DeepSpeed and accelerate configs for two-GPU ZeRO-3;
  - example job configs (replace the `/path/to/...` placeholders with your own paths).

## Apply

```bash
git clone https://github.com/ostris/ai-toolkit.git && cd ai-toolkit
git checkout e03c6e4
git apply /path/to/ai-toolkit-rgba/ai-toolkit-rgba.patch
cp -r /path/to/ai-toolkit-rgba/new_files/. .
```

## What changed

**Required for alpha training and inference**
- **RGBA VAE loading:** load a diffusers `AutoencoderKLFlux2` folder via `vae_path` (`vae_adapter.py`, `flux2_model.py`).
- **RGBA data:** a `rgba: true` dataset option keeps alpha on targets and control images, with alpha bleed under transparent pixels (`dataloader_mixins.py`).
- **RGBA previews:** samples are decoded by the four-channel VAE and saved as transparent PNGs.

**Quality**
- `edge_loss_multiplier`, `edge_loss_dilate`, `edge_loss_source`: weight the loss towards alpha edges.
- `sharpness_loss_multiplier`: match spatial gradients of the predicted clean latent.

**Hardware and convenience**
- Dual-GPU training, and DeepSpeed ZeRO-3 on two GPUs, including a full fine-tune of the 4B transformer.
- Text-embedding cache sentinel and deduplication.
- Web UI job queue and GPU selection.
