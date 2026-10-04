"""FLUX.2 Klein Alpha demo: RGBA foreground extraction on Klein Base 4B/9B and object removal on Base 9B + an RGBA VAE.

Tabs
  Extract         composite + background plate  -> RGBA cut-out            (Extract-4B or Extract-9B LoRA)
  Extract (auto)  photo + brush over the object -> plate (Remove-9B) -> RGBA cut-out (Extract-9B)
  Remove          photo + brush over the object -> photo with the object erased      (Remove-9B LoRA)
  VAE             RGBA PNG -> encode/decode through the 4-channel VAE -> reconstruction + alpha RMSE

The selected 4B or 9B base is loaded (qfloat8, as at training time); both LoRAs are attached as forward hooks on the
quantized Linear layers and switched per request. Inference reuses ai-toolkit's own sampling path
(Flux2Klein9BModel.generate_single_image, with the ai-toolkit-rgba patch applied), which decodes RGBA
latents through the 4-channel VAE.

Usage:  python app.py [--host 127.0.0.1] [--port 7860] [--check-weights]
"""
import argparse
import gc
import os
import json
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from safetensors.torch import load_file

import gradio as gr

# =============================================================================================================
# CONFIGURATION: where the weights come from.
#
# By default everything is downloaded from the Hugging Face repo HF_REPO. Each file can be overridden with an
# environment variable that points to a local copy:
#   VAE_PATH      folder with the diffusers RGBA VAE (config.json + diffusion_pytorch_model.safetensors)
#   EXTRACT_LORA  Extract-9B LoRA .safetensors file
#   EXTRACT_4B_LORA  Extract-4B LoRA .safetensors file
#   REMOVE_LORA   Remove-9B LoRA .safetensors file
#   AITK_PATH     ai-toolkit checkout with the ai-toolkit-rgba patch applied
# =============================================================================================================
HF_REPO = os.environ.get("HF_REPO", "trmz/flux2-klein-alpha")
HF_REVISION = os.environ.get("HF_REVISION") or None  # branch, tag or commit; None = main
HF_VAE_FILES = ("vae/config.json", "vae/diffusion_pytorch_model.safetensors")
HF_EXTRACT_LORA = "loras/extract_9b.safetensors"
HF_EXTRACT_4B_LORA = "loras/extract_4b.safetensors"
HF_REMOVE_LORA = "loras/remove_9b.safetensors"
BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-9B"  # gated: accept its licence on Hugging Face first

# Model settings used when the LoRAs were trained (qfloat8 transformer and text encoder).
MODEL_CONFIG = dict(
    name_or_path=BASE_MODEL, arch="flux2_klein_9b",
    quantize=True, qtype="qfloat8", quantize_te=True, qtype_te="qfloat8",
    low_vram=True, layer_offloading=False, model_kwargs={"match_target_res": False},
)
# Sampling settings for each task, taken from the training job configs.
# scale = linear_alpha / rank (both LoRAs were trained with alpha == rank).
TASKS = {
    "extract": dict(prompt="Foreground", neg="", steps=6, guidance=4.0, seed=42, scale=1.0),
    "remove": dict(prompt="Photo with the object removed, clean background preserved.", neg="", steps=30,
                   guidance=4.0, seed=42, scale=1.0),
}
# =============================================================================================================

PRELOAD = os.environ.get("PRELOAD_9B", "1") == "1"
DEFAULT_EXTRACT_MODEL = os.environ.get("EXTRACT_MODEL", "9B").upper()
if DEFAULT_EXTRACT_MODEL not in ("4B", "9B", "QWEN"):
    raise ValueError("EXTRACT_MODEL must be 4B, 9B or QWEN")
EXTRACT_MODELS = [("FLUX Klein 4B", "4B"), ("FLUX Klein 9B", "9B"), ("Qwen Image 2.1", "QWEN")]
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
EX = os.path.join(HERE, "examples")
MAX_AR = 4  # FLUX.2's reference encoder rejects aspect ratios beyond 4:1
DEV = torch.device("cuda:0")
GPU_LOCK = threading.Lock()  # one GPU job at a time (the queue also enforces this)


# ----------------------------------------------------------------------------------------------- ai-toolkit
def find_aitk():
    """Locate the patched ai-toolkit checkout and put it on sys.path."""
    env = os.environ.get("AITK_PATH")
    candidates = [env] if env else [os.path.join(REPO_ROOT, "ai-toolkit"), os.path.join(os.path.dirname(REPO_ROOT), "ai-toolkit")]
    for c in candidates:
        c = os.path.abspath(os.path.expanduser(c))
        if os.path.isfile(os.path.join(c, "toolkit", "config_modules.py")):
            if not os.path.isfile(os.path.join(c, "extensions_built_in/diffusion_models/flux2/src/vae_adapter.py")):
                sys.exit(f"ai-toolkit found at {c}, but the ai-toolkit-rgba patch is not applied "
                         "(extensions_built_in/diffusion_models/flux2/src/vae_adapter.py is missing).\n"
                         "See ai-toolkit-rgba/README.md.")
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    where = f"AITK_PATH={env}" if env else " or ".join(candidates)
    sys.exit(f"Could not find an ai-toolkit checkout ({where}).\n"
             "Clone https://github.com/ostris/ai-toolkit, apply ai-toolkit-rgba/ai-toolkit-rgba.patch, copy "
             "ai-toolkit-rgba/new_files/, and set AITK_PATH to that folder. See README.md, 'Quick start'.")


# ----------------------------------------------------------------------------------------------- weights
class WeightsError(RuntimeError):
    pass


def _hf_file(filename):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (EntryNotFoundError, GatedRepoError, LocalEntryNotFoundError,
                                        RepositoryNotFoundError, RevisionNotFoundError)
    try:
        return hf_hub_download(HF_REPO, filename, revision=HF_REVISION)
    except GatedRepoError as e:
        raise WeightsError(f"Hugging Face repo '{HF_REPO}' is gated: accept its terms on huggingface.co and log in "
                           f"with `huggingface-cli login`. ({e.__class__.__name__})") from e
    except RepositoryNotFoundError as e:
        raise WeightsError(f"Hugging Face repo '{HF_REPO}' was not found, or it is private and you are not logged "
                           "in with an account that can read it (`huggingface-cli login`). You can also download "
                           "the weights yourself and point VAE_PATH / EXTRACT_LORA / EXTRACT_4B_LORA / REMOVE_LORA at them.") from e
    except RevisionNotFoundError as e:
        raise WeightsError(f"Revision '{HF_REVISION}' does not exist in '{HF_REPO}'.") from e
    except EntryNotFoundError as e:
        raise WeightsError(f"'{filename}' is not in the Hugging Face repo '{HF_REPO}' (yet). Expected layout: "
                           f"{', '.join(HF_VAE_FILES + (HF_EXTRACT_LORA, HF_EXTRACT_4B_LORA, HF_REMOVE_LORA))}. Point the matching "
                           "env var (VAE_PATH / EXTRACT_LORA / EXTRACT_4B_LORA / REMOVE_LORA) at a local copy instead.") from e
    except LocalEntryNotFoundError as e:
        raise WeightsError(f"Could not download '{filename}' from '{HF_REPO}' and it is not in the local cache. "
                           "Check your internet connection, or point the matching env var at a local copy.") from e


def _local(env, kind):
    p = os.path.abspath(os.path.expanduser(os.environ[env]))
    ok = os.path.isdir(p) if kind == "dir" else os.path.isfile(p)
    if not ok:
        raise WeightsError(f"{env}={p} does not exist (expected a {'folder' if kind == 'dir' else 'file'}).")
    return p


def vae_path():
    if os.environ.get("VAE_PATH"):
        p = _local("VAE_PATH", "dir")
        for f in HF_VAE_FILES:
            if not os.path.isfile(os.path.join(p, os.path.basename(f))):
                raise WeightsError(f"VAE_PATH={p} has no {os.path.basename(f)}.")
        return p
    return os.path.dirname([_hf_file(f) for f in HF_VAE_FILES][0])


def lora_path(task, size="9B"):
    if task == "extract" and size == "4B":
        return _local("EXTRACT_4B_LORA", "file") if os.environ.get("EXTRACT_4B_LORA") else _hf_file(HF_EXTRACT_4B_LORA)
    env = {"extract": "EXTRACT_LORA", "remove": "REMOVE_LORA"}[task]
    if os.environ.get(env):
        return _local(env, "file")
    return _hf_file({"extract": HF_EXTRACT_LORA, "remove": HF_REMOVE_LORA}[task])


def describe_sources():
    def src(env, hf):
        return f"local {os.environ[env]}" if os.environ.get(env) else f"hf://{HF_REPO}/{hf}"
    return (f"  VAE:          {src('VAE_PATH', 'vae/')}\n"
            f"  Extract LoRA: {src('EXTRACT_LORA', HF_EXTRACT_LORA)}\n"
            f"  Extract 4B:   {src('EXTRACT_4B_LORA', HF_EXTRACT_4B_LORA)}\n"
            f"  Remove LoRA:  {src('REMOVE_LORA', HF_REMOVE_LORA)}\n"
            f"  9B base:      hf://{BASE_MODEL}\n"
            "  4B base:      hf://black-forest-labs/FLUX.2-klein-base-4B")


# ----------------------------------------------------------------------------------------------- image helpers
def checkerboard(size, cell=16):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    c = (((xx // cell) + (yy // cell)) % 2).astype(np.uint8)
    arr = np.where(c[..., None] == 1, 204, 255).astype(np.uint8).repeat(3, axis=2)
    return Image.fromarray(arr, "RGB")


def over(rgba, bg="checker"):
    rgba = rgba.convert("RGBA")
    base = checkerboard(rgba.size) if bg == "checker" else Image.new("RGB", rgba.size, bg)
    base.paste(rgba, mask=rgba.getchannel("A"))
    return base


def gen_size(w, h, area, multiple=16):
    s = (area / (w * h)) ** 0.5
    return max(multiple, round(w * s / multiple) * multiple), max(multiple, round(h * s / multiple) * multiple)


def pad_to_ar(ims):
    """Edge-pad images (same size) to <= 4:1. Returns padded images, padded size and crop box offset."""
    cw, ch = ims[0].size
    pw, ph = max(cw, -(-ch // MAX_AR)), max(ch, -(-cw // MAX_AR))
    box = ((pw - cw) // 2, (ph - ch) // 2)
    if (pw, ph) == (cw, ch):
        return ims, (pw, ph), box
    out = []
    for im in ims:
        a = np.asarray(im)
        a = np.pad(a, ((box[1], ph - ch - box[1]), (box[0], pw - cw - box[0])) + ((0, 0),) * (a.ndim - 2), mode="edge")
        out.append(Image.fromarray(a))
    return out, (pw, ph), box


def editor_to_photo_and_mask(ed, mask_img=None):
    """Gradio ImageEditor value (+ optional uploaded mask) -> (RGB photo, bool mask of the object).

    An uploaded mask (white = object) takes precedence over the brush layer."""
    if ed is None or ed.get("background") is None:
        raise gr.Error("Upload a photo first.")
    bg = ed["background"]
    bg = Image.fromarray(bg) if isinstance(bg, np.ndarray) else bg
    photo = bg.convert("RGB")
    if mask_img is not None:
        m = mask_img.convert("RGBA")
        a = np.asarray(m.resize(photo.size, Image.NEAREST))
        # white-on-black mask, or a transparent PNG whose opaque pixels mark the object
        mask = (a[..., 3] > 127) if (a[..., 3] < 255).any() else (a[..., :3].mean(-1) > 127)
        if not mask.any():
            raise gr.Error("The uploaded mask is empty (it should be white where the object is).")
        return photo, mask
    mask = np.zeros((photo.height, photo.width), bool)
    for layer in ed.get("layers") or []:
        layer = Image.fromarray(layer) if isinstance(layer, np.ndarray) else layer
        layer = layer.convert("RGBA")
        if layer.size != photo.size:
            layer = layer.resize(photo.size, Image.NEAREST)
        mask |= np.asarray(layer)[..., 3] > 0
    if not mask.any():
        raise gr.Error("Paint over the object with the brush first (or upload a mask).")
    return photo, mask


def dilate(mask, px):
    if px <= 0:
        return mask
    m = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * int(px) + 1))
    return np.asarray(m) > 127


def overlay_mask(photo, mask):
    a = np.asarray(photo.convert("RGB")).astype(np.float32)
    a[mask] = a[mask] * 0.45 + np.array([255, 40, 40]) * 0.55
    return Image.fromarray(a.astype(np.uint8))


def save_png(img, stem):
    d = tempfile.mkdtemp(prefix="f2ka_demo_")
    p = os.path.join(d, f"{stem}.png")
    img.save(p)
    return p


# ----------------------------------------------------------------------------------------------- model engine
class Engine:
    """One Klein base model with LoRAs attached as switchable forward hooks."""

    def __init__(self, size="9B"):
        self.size = size
        self._loading = False
        self.sd = None
        self.pipe = None
        self.embeds = {}
        self.active = None  # name of the active LoRA
        self.loras = {}
        self.load_error = None
        self.load_seconds = None
        self._ready = threading.Event()
        self._calls = 0

    # -- loading
    def load(self):
        self._loading = True
        try:
            self._load()
        except Exception as e:  # surface in the UI instead of dying silently
            self.load_error = str(e) if isinstance(e, WeightsError) else repr(e)
            raise
        finally:
            self._ready.set()

    def _load(self):
        find_aitk()
        from toolkit.config_modules import ModelConfig
        from extensions_built_in.diffusion_models.flux2.flux2_klein_model import Flux2Klein9BModel, Flux2Klein4BModel
        from extensions_built_in.diffusion_models.flux2.flux2_model import Flux2Model

        t0 = time.time()
        # resolve (and download, if needed) the small files first, so a missing file fails before the base model loads
        vae = vae_path()
        tasks = TASKS if self.size == "9B" else {"extract": dict(TASKS["extract"], steps=6)}
        self.cfg = {task: dict(c, lora=lora_path(task, self.size)) for task, c in tasks.items()}
        model_config = dict(MODEL_CONFIG)
        model_config.update(name_or_path=f"black-forest-labs/FLUX.2-klein-base-{self.size}",
                            arch=f"flux2_klein_{self.size.lower()}")
        model_class = Flux2Klein4BModel if self.size == "4B" else Flux2Klein9BModel
        sd = model_class(device="cuda:0", model_config=ModelConfig(**model_config, vae_path=vae), dtype="bf16",
                               noise_scheduler=Flux2Model.get_train_scheduler())
        sd.load_model()
        pipe = sd.pipeline
        type(pipe)._execution_device = property(lambda self: DEV)
        assert sd.is_rgba_vae, "VAE is not 4-channel"
        te = pipe.text_encoder
        te.to(DEV)
        with torch.no_grad():
            for k, c in self.cfg.items():
                self.embeds[k] = (sd.get_prompt_embeds(c["prompt"]), sd.get_prompt_embeds(c["neg"]))
        te.to("cpu")
        torch.cuda.empty_cache()

        mods = dict(pipe.transformer.named_modules())
        hooked = set()
        for name, c in self.cfg.items():
            sdict = load_file(c["lora"])
            n = 0
            for k in [k for k in sdict if k.endswith(".lora_A.weight")]:
                mname = k[len("diffusion_model."):-len(".lora_A.weight")]
                mod = mods[mname]
                A = sdict[k].to(DEV, torch.bfloat16)
                B = sdict[k.replace("lora_A", "lora_B")].to(DEV, torch.bfloat16)
                if not hasattr(mod, "_loras"):
                    mod._loras = {}
                mod._loras[name] = (A, B, c["scale"])
                if mname not in hooked:
                    mod.register_forward_hook(self._hook)
                    hooked.add(mname)
                n += 1
            self.loras[name] = n
        pipe.transformer.register_forward_pre_hook(self._count)
        pipe.transformer.to(DEV)
        sd.vae.to(DEV)
        self.sd, self.pipe = sd, pipe
        self.load_seconds = time.time() - t0
        print(f"[engine] {self.size} loaded in {self.load_seconds:.0f}s, LoRA modules {self.loras}", flush=True)

    def _hook(self, m, inp, out):
        lo = m._loras.get(self.active)
        if lo is None:
            return out
        A, B, s = lo
        x = inp[0].to(A.dtype)
        return out + (x @ A.t() @ B.t()).to(out.dtype) * s

    def _count(self, m, args, kwargs=None):
        self._calls += 1

    def wait(self, progress=None):
        if not self._ready.is_set():
            if progress is not None:
                label = "Qwen Image 2.1" if self.size == "QWEN" else f"FLUX.2 Klein {self.size}"
                progress(0, desc=f"Loading {label}...")
            if self.sd is None and not self._loading:
                self._loading = True
                self.load()
            self._ready.wait()
        if self.load_error:
            raise gr.Error(f"Model failed to load: {self.load_error}")

    def unload(self):
        if self.pipe is not None:
            # Quantization caches can retain modules after their Python owner is
            # released. Move their storage off GPU before loading another base.
            for mod in self.pipe.transformer.modules():
                if hasattr(mod, "_loras"):
                    mod._loras.clear()
            self.pipe.transformer.to("cpu")
            self.pipe.text_encoder.to("cpu")
            self.sd.vae.to("cpu")
        self.sd = self.pipe = None
        self.embeds.clear()
        self.cfg = {}
        self._ready.clear()
        gc.collect()
        torch.cuda.empty_cache()

    # -- generation
    def generate(self, task, controls, size, progress=None, desc="", seed=None, steps=None):
        """controls: list of PIL images already at `size`. Returns an RGBA PIL image at `size`."""
        from toolkit.config_modules import GenerateImageConfig

        c = self.cfg[task]
        steps = int(steps or c["steps"])
        seed = int(c["seed"] if seed is None else seed)
        W, H = size
        tmp = tempfile.mkdtemp(prefix="f2ka_ctl_")
        paths = []
        for i, im in enumerate(controls):
            p = os.path.join(tmp, f"c{i + 1}.png")
            im.save(p)
            paths.append(p)
        g = GenerateImageConfig(prompt=c["prompt"], width=W, height=H, num_inference_steps=steps,
                                guidance_scale=c["guidance"], seed=seed, output_folder=tmp, output_ext="png",
                                ctrl_img_1=paths[0], ctrl_img_2=paths[1] if len(paths) > 1 else None)
        pos, neg = self.embeds[task]
        gen = torch.Generator(device=DEV).manual_seed(seed)
        total = steps * (2 if c["guidance"] > 1 else 1)
        stop = threading.Event()

        def ticker():  # the pipeline has no step callback; count transformer calls instead
            while not stop.wait(0.5):
                if progress is not None:
                    k = min(self._calls, total)
                    progress(k / total, desc=f"{desc} step {k * steps // total}/{steps}")

        with GPU_LOCK:
            self.active = task
            self._calls = 0
            th = threading.Thread(target=ticker, daemon=True)
            th.start()
            try:
                with torch.no_grad():
                    img = self.sd.generate_single_image(self.pipe, g, pos, neg, gen, {})
            finally:
                stop.set()
                th.join()
                self.active = None
                for p in paths:
                    os.remove(p)
                os.rmdir(tmp)
        return img


class QwenEngine(Engine):
    """Persistent Qwen worker in its own Python environment, on the demo GPU."""
    def __init__(self):
        super().__init__("QWEN")
        self.worker = None
        self.worker_log = None
        self.peak_vram = 0

    def _request(self, command, progress=None):
        self.worker.stdin.write(json.dumps(command) + "\n")
        self.worker.stdin.flush()
        while True:
            line = self.worker.stdout.readline()
            if not line:
                raise RuntimeError(f"Qwen worker exited. See {self.worker_log.name}")
            response = json.loads(line)
            if "progress" in response:
                if progress is not None:
                    progress(response["progress"], desc=response["desc"])
                continue
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "Qwen worker failed"))
            return response

    def _load(self):
        python = os.environ.get("QWEN_PYTHON", sys.executable)
        self.worker_log = tempfile.NamedTemporaryFile(prefix="plate_extract_worker_", suffix=".log", mode="w", delete=False)
        self.worker = subprocess.Popen([python, "-u", os.path.join(HERE, "qwen_extract.py"), "--worker"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=self.worker_log, text=True, bufsize=1)
        try:
            response = self._request({"action": "load"})
        except Exception:
            self.worker.terminate()
            self.worker.wait()
            raise
        self.sd = self.worker
        self.loras = {"extract": response["modules"]}
        self.load_seconds = response["load_seconds"]

    def generate(self, task, controls, size, progress=None, desc="", seed=None, steps=None):
        if task != "extract":
            raise ValueError("Qwen supports extraction; removal uses FLUX Klein 9B")
        with GPU_LOCK, tempfile.TemporaryDirectory(prefix="plate_extract_") as folder:
            comp, bg, output = [os.path.join(folder, name) for name in ("composite.png", "plate.png", "output.png")]
            controls[0].save(comp)
            controls[1].save(bg)
            response = self._request(dict(action="extract", composite=comp, background=bg, output=output,
                                          width=size[0], height=size[1], seed=0 if seed is None else int(seed)), progress)
            self.peak_vram = response["peak_vram_gib"]
            with Image.open(output) as image:
                return image.copy()

    def unload(self):
        if self.worker is not None and self.worker.poll() is None:
            try:
                self._request({"action": "unload"})
                self.worker.wait(timeout=30)
            except Exception:
                self.worker.kill()
                self.worker.wait()
        if self.worker is not None:
            self.worker.stdin.close()
            self.worker.stdout.close()
        if self.worker_log is not None:
            self.worker_log.close()
        self.worker = self.sd = self.pipe = None
        self._ready.clear()


def new_engine(model):
    return QwenEngine() if model == "QWEN" else Engine(model)


ENGINE = new_engine(DEFAULT_EXTRACT_MODEL)


def get_engine(size, progress=None):
    """Keep only one base resident when switching between FLUX and Qwen."""
    global ENGINE
    if ENGINE.size != size:
        if ENGINE._loading and not ENGINE._ready.is_set():
            ENGINE._ready.wait()
        with GPU_LOCK:
            ENGINE.unload()
            ENGINE = new_engine(size)
    ENGINE.wait(progress)
    return ENGINE


# ----------------------------------------------------------------------------------------------- VAE (tab 3)
class VAE:
    def __init__(self):
        self.vae = None
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            if self.vae is None:
                from diffusers import AutoencoderKLFlux2
                self.vae = AutoencoderKLFlux2.from_pretrained(vae_path(), torch_dtype=torch.bfloat16).to(DEV).eval()
        return self.vae


VAE_ = VAE()


# ----------------------------------------------------------------------------------------------- task functions
def _vram():
    return max(torch.cuda.max_memory_allocated(DEV) / 2**30,
               ENGINE.peak_vram if isinstance(ENGINE, QwenEngine) else 0)


def run_extract_core(comp, plate, area, progress, desc="Extracting", model="9B"):
    engine = get_engine(model, progress)
    comp = comp.convert("RGB")
    plate = plate.convert("RGB").resize(comp.size, Image.BICUBIC)
    (comp_in, plate_in), (pw, ph), box = pad_to_ar([comp, plate])
    W, H = gen_size(pw, ph, area, multiple=32 if model == "QWEN" else 16)
    out = engine.generate("extract", [comp_in.resize((W, H), Image.BICUBIC), plate_in.resize((W, H), Image.BICUBIC)],
                          (W, H), progress, desc)
    cw, ch = comp.size
    out = out.resize((pw, ph), Image.LANCZOS).crop((box[0], box[1], box[0] + cw, box[1] + ch))
    return out, (W, H)


def run_remove_core(photo, mask, area, grow, keep_outside, progress, desc="Removing"):
    engine = get_engine("9B", progress)
    mask = dilate(mask, grow)
    alpha = np.where(mask, 128, 255).astype(np.uint8)
    ctrl = Image.fromarray(np.dstack([np.asarray(photo), alpha]), "RGBA")
    (ctrl_in,), (pw, ph), box = pad_to_ar([ctrl])
    W, H = gen_size(pw, ph, area)
    out = engine.generate("remove", [ctrl_in.resize((W, H), Image.BICUBIC)], (W, H), progress, desc)
    cw, ch = photo.size
    out = out.convert("RGB").resize((pw, ph), Image.LANCZOS).crop((box[0], box[1], box[0] + cw, box[1] + ch))
    if keep_outside:
        # keep the original pixels away from the brushed region; feathered blend at the border
        soft = Image.fromarray(dilate(mask, max(8, cw // 64)).astype(np.uint8) * 255).filter(
            ImageFilter.GaussianBlur(max(3, cw // 160)))
        out = Image.composite(out, photo, soft)
    return out, (W, H), mask


AREAS = {"512² (fast)": 512 * 512, "768² (~2x slower)": 768 * 768, "1024² (slow)": 1024 * 1024}


def tab_extract(comp, plate, res, model="9B", progress=gr.Progress()):
    if comp is None or plate is None:
        raise gr.Error("Upload both the composite and the background plate.")
    get_engine(model, progress)
    torch.cuda.reset_peak_memory_stats(DEV)
    t = time.time()
    rgba, gs = run_extract_core(comp, plate, AREAS[res], progress, model=model)
    dt = time.time() - t
    a = np.asarray(rgba)[..., 3]
    label = "PlateExtract (Qwen)" if model == "QWEN" else f"Extract-{model}"
    status = (f"{label}: done in {dt:.1f} s at generation size {gs[0]}x{gs[1]} (peak VRAM {_vram():.1f} GB). "
              f"Opaque pixels: {100 * (a > 127).mean():.1f}%.")
    return over(rgba), rgba.getchannel("A"), save_png(rgba, "cutout"), status


def tab_remove(ed, mask_img, res, grow, keep_outside, progress=gr.Progress()):
    photo, mask = editor_to_photo_and_mask(ed, mask_img)
    get_engine("9B", progress)
    torch.cuda.reset_peak_memory_stats(DEV)
    t = time.time()
    out, gs, used = run_remove_core(photo, mask, AREAS[res], grow, keep_outside, progress)
    dt = time.time() - t
    status = f"Done in {dt:.1f} s at generation size {gs[0]}x{gs[1]} (peak VRAM {_vram():.1f} GB)."
    return overlay_mask(photo, used), out, save_png(out, "removed"), status


def tab_auto(ed, mask_img, res, grow, model="9B", progress=gr.Progress()):
    photo, mask = editor_to_photo_and_mask(ed, mask_img)
    get_engine("9B", progress)
    torch.cuda.reset_peak_memory_stats(DEV)
    t = time.time()
    plate, gs, used = run_remove_core(photo, mask, AREAS[res], grow, True, progress, "1/2 Generating background plate:")
    t1 = time.time() - t
    rgba, _ = run_extract_core(photo, plate, AREAS[res], progress, "2/2 Extracting:", model=model)
    dt = time.time() - t
    status = (f"Done in {dt:.1f} s (plate {t1:.1f} s + extract {dt - t1:.1f} s) at {gs[0]}x{gs[1]}, "
              f"peak VRAM {_vram():.1f} GB.")
    return overlay_mask(photo, used), plate, over(rgba), rgba.getchannel("A"), save_png(rgba, "cutout_auto"), status


def _label(im, text):
    from PIL import ImageFont
    fs = max(12, min(im.size) // 22)
    font = ImageFont.load_default(size=fs)
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = d.textbbox((6, 4), text, font=font)
    d.rectangle([0, 0, x1 + 6, y1 + 4], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 255), font=font)
    return im


def tab_vae(img, progress=gr.Progress()):
    if img is None:
        raise gr.Error("Upload an RGBA PNG.")
    img = img.convert("RGBA")
    w0, h0 = img.size
    s = min(1.0, 1536 / max(w0, h0))  # keep memory bounded
    if s < 1:
        img = img.resize((round(w0 * s), round(h0 * s)), Image.LANCZOS)
    w, h = img.size
    W, H = -(-w // 16) * 16, -(-h // 16) * 16
    progress(0.1, desc="Encoding / decoding")
    arr = np.asarray(img)
    arr_p = np.pad(arr, ((0, H - h), (0, W - w), (0, 0)), mode="edge")
    try:
        vae = VAE_.get()
    except WeightsError as e:
        raise gr.Error(str(e))
    t = time.time()
    with GPU_LOCK, torch.no_grad():
        x = torch.from_numpy(arr_p).permute(2, 0, 1)[None].to(DEV, torch.bfloat16) / 127.5 - 1
        z = vae.encode(x).latent_dist.mode()
        y = vae.decode(z).sample.float().clamp(-1, 1)
    torch.cuda.synchronize(DEV)
    dt = time.time() - t
    rec = ((y[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)[:h, :w]
    rec_img = Image.fromarray(rec, "RGBA")
    a0, a1 = arr[..., 3].astype(np.float64) / 255, rec[..., 3].astype(np.float64) / 255
    rmse = np.sqrt(((a0 - a1) ** 2).mean())
    pm0 = arr[..., :3] / 255 * a0[..., None]
    pm1 = rec[..., :3] / 255 * a1[..., None]
    mse = ((pm0 - pm1) ** 2).mean()
    psnr = 10 * np.log10(1 / max(mse, 1e-12))
    rows = []
    for name, im in (("original", img), ("reconstruction", rec_img)):
        tiles = [_label(over(im, bg), f"{name} / {lab}") for bg, lab in
                 (("checker", "checker"), ((0, 0, 0), "black"), ((255, 255, 255), "white"))]
        rows.append(tiles)
    grid = Image.new("RGB", (3 * w + 8, 2 * h + 4), (128, 128, 128))
    for r, tiles in enumerate(rows):
        for c, tile in enumerate(tiles):
            grid.paste(tile, (c * (w + 4), r * (h + 4)))
    status = (f"Alpha RMSE: {rmse:.4f} (0-1 scale, = {rmse * 255:.2f}/255). "
              f"Premultiplied-RGB PSNR: {psnr:.2f} dB. Latent {tuple(z.shape[1:])}. "
              f"Size {w}x{h}{' (downscaled from %dx%d)' % (w0, h0) if s < 1 else ''}. {dt * 1000:.0f} ms on GPU.")
    return grid, rec_img.getchannel("A"), save_png(rec_img, "reconstruction"), status


# ----------------------------------------------------------------------------------------------- UI
BRUSH = gr.Brush(colors=["#ff3030"], color_mode="fixed", default_size=40)


def mask_editor(label):
    return gr.ImageEditor(label=label, type="pil", image_mode="RGBA", brush=BRUSH, eraser=gr.Eraser(),
                          layers=False, transforms=(), sources=("upload", "clipboard"), height=520)


def mask_upload():
    with gr.Accordion("…or upload a mask instead of brushing (white = object)", open=False):
        return gr.Image(label="Mask (optional; overrides the brush)", type="pil", image_mode="RGBA", height=200)


def ex_editor(photo):
    # Examples ship a mask image rather than a pre-painted brush layer: Gradio 6.3's ImageEditor places
    # pre-loaded layers at an offset, so the mask would not line up with the photo.
    return {"background": os.path.join(EX, photo), "layers": [], "composite": None}


INTRO = """# RGBA extraction — FLUX Klein 4B / 9B or Qwen Image 2.1
Choose an extractor to get a transparent PNG. Qwen uses its native RGBA VAE; object removal uses FLUX Remove-9B.
Built with Qwen. Generation runs on one GPU, one request at a time. Switching extractors reloads the model.
"""

with gr.Blocks(title="FLUX.2 Klein Alpha demo") as demo:
    gr.Markdown(INTRO)
    with gr.Tabs():
        # ---- Extract
        with gr.Tab("Extract"):
            gr.Markdown("**FLUX 4B / FLUX 9B / Qwen.** Give the picture with the object (*composite*) and the same picture "
                        "**without** the object (*background plate*). Returns the object as a transparent RGBA PNG.")
            with gr.Row():
                with gr.Column():
                    ex_comp = gr.Image(label="Composite (object on background)", type="pil", image_mode="RGB", height=320)
                    ex_plate = gr.Image(label="Background plate (same background, no object)", type="pil",
                                        image_mode="RGB", height=320)
                    ex_res = gr.Radio(list(AREAS), value=list(AREAS)[0], label="Generation size (pixel area)")
                    ex_model = gr.Radio(EXTRACT_MODELS, value=DEFAULT_EXTRACT_MODEL, label="Extractor model")
                    ex_btn = gr.Button("Extract", variant="primary")
                with gr.Column():
                    ex_out = gr.Image(label="Cut-out on checkerboard", type="pil", format="png", height=380)
                    with gr.Row():
                        ex_alpha = gr.Image(label="Alpha matte", type="pil", height=200)
                        ex_file = gr.File(label="Download RGBA PNG")
                    ex_status = gr.Markdown()
            gr.Examples([[os.path.join(EX, f"extract_{n}_composite.png"), os.path.join(EX, f"extract_{n}_plate.png")]
                         for n in ("watercolor", "anime")], [ex_comp, ex_plate], label="Examples (benchmark holdout)")
            ex_btn.click(tab_extract, [ex_comp, ex_plate, ex_res, ex_model], [ex_out, ex_alpha, ex_file, ex_status],
                         concurrency_id="gpu")

        # ---- Extract (auto)
        with gr.Tab("Extract (auto background)"):
            gr.Markdown("**Remove-9B, then your chosen extractor.** No plate needed: brush over the object, the remover "
                        "paints the background plate, and the extractor cuts the object out against it. "
                        "Takes two generations. Choosing 4B or Qwen reloads the model between stages. Quality depends on the remover (see README).")
            with gr.Row():
                with gr.Column():
                    au_ed = mask_editor("Photo — brush over the object to cut out")
                    au_mask_in = mask_upload()
                    au_res = gr.Radio(list(AREAS), value=list(AREAS)[0], label="Generation size (pixel area)")
                    au_model = gr.Radio(EXTRACT_MODELS, value=DEFAULT_EXTRACT_MODEL, label="Extractor model")
                    au_grow = gr.Slider(0, 40, value=8, step=1, label="Grow brushed mask (px)")
                    au_btn = gr.Button("Extract with generated plate", variant="primary")
                with gr.Column():
                    with gr.Row():
                        au_mask = gr.Image(label="Mask used (red)", type="pil", height=240)
                        au_plate = gr.Image(label="Generated background plate", type="pil", height=240)
                    au_out = gr.Image(label="Cut-out on checkerboard", type="pil", format="png", height=320)
                    with gr.Row():
                        au_alpha = gr.Image(label="Alpha matte", type="pil", height=180)
                        au_file = gr.File(label="Download RGBA PNG")
                    au_status = gr.Markdown()
            gr.Examples([[ex_editor("extract_anime_composite.png"), os.path.join(EX, "auto_anime_mask.png")]],
                        [au_ed, au_mask_in], label="Example (benchmark holdout, with mask)")
            au_btn.click(tab_auto, [au_ed, au_mask_in, au_res, au_grow, au_model], [au_mask, au_plate, au_out, au_alpha, au_file, au_status],
                         concurrency_id="gpu")

        # ---- Remove
        with gr.Tab("Remove"):
            gr.Markdown("**Remove-9B.** Upload a photo, paint over the object with the brush, and the model erases "
                        "it and fills in the background (including its shadow, when it works).")
            with gr.Row():
                with gr.Column():
                    rm_ed = mask_editor("Photo — brush over the object to remove")
                    rm_mask_in = mask_upload()
                    rm_res = gr.Radio(list(AREAS), value=list(AREAS)[0], label="Generation size (pixel area)")
                    rm_grow = gr.Slider(0, 40, value=4, step=1, label="Grow brushed mask (px)")
                    rm_keep = gr.Checkbox(False, label="Only change the brushed area (keeps original full-res pixels elsewhere, but the object's shadow/reflection stays)")
                    rm_btn = gr.Button("Remove object", variant="primary")
                with gr.Column():
                    rm_mask = gr.Image(label="Mask used (red)", type="pil", height=260)
                    rm_out = gr.Image(label="Result", type="pil", format="png", height=380)
                    rm_file = gr.File(label="Download PNG")
                    rm_status = gr.Markdown()
            gr.Examples([[ex_editor("remove_photo.png"), os.path.join(EX, "remove_mask.png")]], [rm_ed, rm_mask_in],
                        label="Example (training photo, with mask)")
            rm_btn.click(tab_remove, [rm_ed, rm_mask_in, rm_res, rm_grow, rm_keep], [rm_mask, rm_out, rm_file, rm_status],
                         concurrency_id="gpu")

        # ---- VAE
        with gr.Tab("VAE encode/decode"):
            gr.Markdown("**RGBA VAE only** — no diffusion model. Upload a PNG with transparency; "
                        "it is encoded to a 32-channel latent and decoded back. Top row: original, bottom row: "
                        "reconstruction, on checkerboard / black / white.")
            with gr.Row():
                with gr.Column(scale=1):
                    va_in = gr.Image(label="RGBA PNG", type="pil", image_mode="RGBA", format="png", height=320)
                    va_btn = gr.Button("Encode + decode", variant="primary")
                    va_status = gr.Markdown()
                    va_file = gr.File(label="Download reconstruction PNG")
                with gr.Column(scale=2):
                    va_grid = gr.Image(label="Original (top) vs reconstruction (bottom)", type="pil", height=480)
                    va_alpha = gr.Image(label="Reconstructed alpha", type="pil", height=200)
            gr.Examples([[os.path.join(EX, f"extract_{n}_groundtruth.png")] for n in ("watercolor", "anime")],
                        [va_in], label="Examples (ground-truth cut-outs)")
            va_btn.click(tab_vae, [va_in], [va_grid, va_alpha, va_file, va_status], concurrency_id="vae")


def main():
    ap = argparse.ArgumentParser(description="FLUX.2 Klein Alpha Gradio demo")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                    help="interface to listen on (default 127.0.0.1; use 0.0.0.0 to expose it on your network)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--check-weights", action="store_true",
                    help="resolve/download the weights, print their paths and exit (no GPU needed)")
    args = ap.parse_args()

    print("Weights:\n" + describe_sources(), flush=True)
    if args.check_weights:
        try:
            print("VAE folder:   ", vae_path())
            for t in TASKS:
                print(f"{t} LoRA:".ljust(14), lora_path(t))
            print("extract 4B:", lora_path("extract", "4B"))
            from qwen_extract import lora_path as qwen_lora_path
            print("extract Qwen:", qwen_lora_path())
        except WeightsError as e:
            sys.exit(f"ERROR: {e}")
        return

    if DEFAULT_EXTRACT_MODEL != "QWEN":
        aitk = find_aitk()
        print(f"ai-toolkit:    {aitk}", flush=True)
    if not torch.cuda.is_available():
        sys.exit("A CUDA GPU is required.")
    if PRELOAD:
        ENGINE._loading = True
        threading.Thread(target=ENGINE.load, daemon=True).start()
    demo.queue(default_concurrency_limit=1, max_size=20)
    demo.launch(server_name=args.host, server_port=args.port, share=False, show_error=True,
                theme=gr.themes.Soft(), allowed_paths=[EX])


if __name__ == "__main__":
    main()
