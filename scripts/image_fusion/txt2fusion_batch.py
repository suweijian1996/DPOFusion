import argparse, os, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from imwatermark import WatermarkEncoder
from itertools import islice
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast

# ===== 原脚本里的工具函数 =====
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

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
    model.cuda(); model.eval()
    return model

def load_and_preprocess(path, H, W):
    img = Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)
    x = np.array(img).astype(np.float32) / 255.0
    x = x * 2.0 - 1.0
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).contiguous()
    return x

# ===== 新增：文件批处理辅助 =====
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(d):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(d, f"**/*{ext}"), recursive=True))
        paths.extend(glob.glob(os.path.join(d, f"*{ext}"), recursive=False))
    # 去重并稳定排序
    paths = sorted(set(paths))
    return paths

def make_out_path(out_root, rel_path):
    dst = os.path.join(out_root, rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    return dst


def process_pair(model, device, ir_path, vis_path, save_path, H, W, prompt, ddim_steps, ddim_eta, precision):
    """对一对 IR/VIS 文件做一次融合并保存到 save_path"""
    with torch.no_grad():
        precision_scope = autocast if precision == "autocast" else nullcontext
        with precision_scope():
            with model.ema_scope():
                # 1) 读入与预处理
                x_ir  = load_and_preprocess(ir_path,  H, W).to(device)
                x_vis = load_and_preprocess(vis_path, H, W).to(device)

                # 2) 文本条件（cross-attn）
                if model.cond_stage_model is not None:
                    c = model.get_learned_conditioning([prompt])
                else:
                    c = None

                # 3) 编码到潜空间并拼接
                enc_ir  = model.encode_first_stage(x_ir)
                enc_vis = model.encode_first_stage(x_vis)
                z_ir  = model.get_first_stage_encoding(enc_ir)
                z_vis = model.get_first_stage_encoding(enc_vis)
                z_code = torch.cat([z_ir, z_vis], dim=1)  # (B,6,h,w)

                # 4) 采样
                samples_z, _ = model.sample_log(
                    z_code=z_code, cond=c, batch_size=1,
                    ddim=True, ddim_steps=ddim_steps, eta=ddim_eta
                )

                # 5) 解码与保存（输出单通道），加水印后保存为 PNG（RGB）
                x = model.decode_first_stage(samples_z)
                x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
                img_np = (x[0, 0].cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_np, mode="L").convert("RGB")
                # img = put_watermark(img, wm_encoder)
                img.save(save_path)


def main():
    parser = argparse.ArgumentParser()

    # ======= 关键改动：支持目录批处理 =======
    parser.add_argument("--fusion", action="store_true", help="use ddpm_fusion pipeline (IR+VIS)")
    parser.add_argument("--ir", type=str, help="path to single IR image")
    parser.add_argument("--vis", type=str, help="path to single VIS image")
    parser.add_argument("--ir_dir", type=str, help="folder of IR images (batch mode)")
    parser.add_argument("--vis_dir", type=str, help="folder of VIS images (batch mode)")
    parser.add_argument("--pair_by", type=str, default="stem", choices=["stem", "name"],
                        help="如何配对：stem=不含扩展名的文件名相同；name=完整文件名相同")

    # 其余参数保持与原脚本一致（择要列出与批处理相关的）
    parser.add_argument("--prompt", type=str, default="The fused image with full infrared information and no visible information")
    parser.add_argument("--outdir", type=str, default="./outputs/fused_s1/")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--config", type=str, default="./configs/fusion_anything/ldm-txt2fusion.yaml")
    parser.add_argument("--ckpt", type=str, default="./models/ldm_fusion_model.ckpt")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--precision", type=str, choices=["full", "autocast"], default="autocast")
    parser.add_argument("--plms", action='store_true')
    parser.add_argument("--dpm_solver", action='store_true')

    opt = parser.parse_args()

    # ====== 加载模型 ======
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # 采样器（非 fusion 路径才用，这里保持兼容）
    if opt.dpm_solver:
        _ = DPMSolverSampler(model)
    elif opt.plms:
        _ = PLMSSampler(model)
    else:
        _ = DDIMSampler(model)

    # 输出目录
    os.makedirs(opt.outdir, exist_ok=True)
    sample_path = os.path.join(opt.outdir)
    os.makedirs(sample_path, exist_ok=True)

    # 水印
    # wm_encoder = WatermarkEncoder(); wm_encoder.set_watermark('bytes', b"StableDiffusionV1")

    # ======= 单张 or 批处理 =======
    if not opt.fusion:
        raise SystemExit("This patch focuses on --fusion(IR+VIS) 模式的批处理。请加 --fusion。")

    # ---- 单张模式兼容 ----
    if opt.ir and opt.vis:
        # 保存为 outdir/samples/00000.png
        base_count = len([p for p in os.listdir(sample_path) if p.lower().endswith('.png')])
        save_path = os.path.join(sample_path, f"{base_count:05}.png")
        process_pair(model, device, opt.ir, opt.vis, save_path,
                     opt.H, opt.W, opt.prompt, opt.ddim_steps, opt.ddim_eta, opt.precision)
        print(f"[fusion] saved: {save_path}")
        print(f"Done. Folder: {opt.outdir}")
        return

    # ---- 批处理模式 ----
    assert opt.ir_dir and opt.vis_dir, "batch 模式需要同时提供 --ir_dir 与 --vis_dir"

    ir_paths  = list_images(opt.ir_dir)
    vis_paths = list_images(opt.vis_dir)

    # 根据文件名或 stem 建索引
    def key_fn(p):
        name = os.path.basename(p)
        return os.path.splitext(name)[0] if opt.pair_by == "stem" else name

    vis_index = {key_fn(p): p for p in vis_paths}

    # 在 outdir 下复刻 ir_dir 的目录层级，把扩展名统一存为 .png
    base_ir_root = os.path.abspath(opt.ir_dir)
    base_count = len([p for p in os.listdir(sample_path) if p.lower().endswith('.png')])

    paired, skipped = 0, 0
    for ir_p in tqdm(ir_paths, desc="Fusing batches"):
        k = key_fn(ir_p)
        vis_p = vis_index.get(k)
        if vis_p is None:
            skipped += 1
            continue
        # 生成相对路径
        rel = os.path.relpath(ir_p, base_ir_root)
        rel_noext = os.path.splitext(rel)[0] + ".png"
        save_p = make_out_path(sample_path, rel_noext)

        process_pair(model, device, ir_p, vis_p, save_p,
                     opt.H, opt.W, opt.prompt, opt.ddim_steps, opt.ddim_eta, opt.precision)
        paired += 1
        base_count += 1

    print(f"[fusion] paired={paired}, skipped(no match)={skipped}")
    print(f"All done. Results at: {opt.outdir}")


if __name__ == "__main__":
    main()
