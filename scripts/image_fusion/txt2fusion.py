import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# from transformers import AutoFeatureExtractor


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        # 确保是 RGB（灰度会导致 cv2.cvtColor 报错）
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def load_and_preprocess(path, H, W):
    img = Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)
    #  repeat grayscale to RGB
    x = np.array(img).astype(np.float32) / 255.0
    x = x * 2.0 - 1.0                    # [-1, 1]
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).contiguous()
    return x


def main():
    parser = argparse.ArgumentParser()

    # ===== ddpm_fusion 相关 =====
    parser.add_argument(
        "--fusion",
        action="store_true",
        help="use ddpm_fusion pipeline (IR+VIS)"
    )
    parser.add_argument("--ir", type=str, help="path to IR image (single file)")
    parser.add_argument("--vis", type=str, help="path to VIS image (single file)")

    # ===== 其余参数（沿用原脚本） =====
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="The fused image",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=9,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=4,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model = model.to(device)

    # ========== 采样器（仅非 fusion 路径会用到） ==========
    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    # ========== 输出目录 ==========
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext

    # =========================
    # FUSION: ddpm_fusion 推理
    # =========================
    if opt.fusion:
        assert opt.ir and opt.vis, "fusion 模式需要同时提供 --ir 与 --vis"
        with torch.no_grad():
            with precision_scope():
                with model.ema_scope():
                    # 1) 读入与预处理
                    x_ir  = load_and_preprocess(opt.ir,  opt.H, opt.W).to(device)
                    x_vis = load_and_preprocess(opt.vis, opt.H, opt.W).to(device)

                    # 2) 文本条件（cross-attn）
                    if model.cond_stage_model is not None:
                        print(f"[fusion] using text condition: {opt.prompt}")
                        c = model.get_learned_conditioning([opt.prompt])
                    else:
                        c = None

                    # 3) 编码到潜空间并拼接成 z_code
                    enc_ir  = model.encode_first_stage(x_ir)
                    enc_vis = model.encode_first_stage(x_vis)
                    z_ir  = model.get_first_stage_encoding(enc_ir)
                    z_vis = model.get_first_stage_encoding(enc_vis)
                    z_code = torch.cat([z_ir, z_vis], dim=1)  # (B,6,h,w)

                    # 4) 采样（注意：用 model.sample_log，内部会调用带 z_code 的 DDIM 分支）
                    samples_z, _ = model.sample_log(
                        z_code=z_code,
                        cond=c,
                        batch_size=1,
                        ddim=True,
                        ddim_steps=opt.ddim_steps,
                        eta=opt.ddim_eta
                    )

                    # 5) 解码与保存（输出是单通道）
                    x = model.decode_first_stage(samples_z)  # (B,1,H,W)
                    x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
                    img_np = (x[0, 0].cpu().numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(img_np, mode="L")
                    # put watermark 需要 RGB，这里转一下再水印
                    img_rgb = img.convert("RGB")
                    # img_rgb = put_watermark(img_rgb, wm_encoder)
                    img_rgb.save(os.path.join(sample_path, f"{base_count:05}.png"))
                    print(f"[fusion] saved: {os.path.join(sample_path, f'{base_count:05}.png')}")
                    base_count += 1

        print(f"Your fusion sample is ready here:\n{outpath}\n\nEnjoy.")
        return

if __name__ == "__main__":
    main()
