import argparse, os, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast

# ----------------- 基础工具 -----------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(d):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(d, f"**/*{ext}"), recursive=True))
        paths.extend(glob.glob(os.path.join(d, f"*{ext}"), recursive=False))
    return sorted(set(paths))

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose: print("missing keys:", m)
    if len(u) > 0 and verbose: print("unexpected keys:", u)
    model.cuda(); model.eval()
    return model

def pil_to_tensor_nchw(img: Image.Image):
    x = np.array(img).astype(np.float32) / 255.0
    x = x * 2.0 - 1.0
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).contiguous()
    return x

def tensor_to_gray_pil(x: torch.Tensor) -> Image.Image:
    # x: (1,C,h,w) in [-1,1]
    x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    if x.shape[1] == 1:
        arr = (x[0,0].detach().cpu().numpy()*255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    else:
        arr = (x[0,0].detach().cpu().numpy()*255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

def load_pil(path): return Image.open(path).convert("RGB")

def intersect_size(a_wh, b_wh):
    return min(a_wh[0], b_wh[0]), min(a_wh[1], b_wh[1])

def choose_patch_box(W, H, pw, ph, mode="topleft", px=None, py=None):
    """在 W×H 的区域中选择一个 pw×ph 的 patch 框 (x0,y0,x1,y1)"""
    if pw > W or ph > H:
        raise ValueError(f"Patch ({pw}x{ph}) larger than overlap ({W}x{H})")
    if mode == "topleft":
        x0, y0 = 0, 0
    elif mode == "center":
        x0 = (W - pw) // 2
        y0 = (H - ph) // 2
    elif mode == "xy":
        x0 = int(np.clip(px if px is not None else 0, 0, W - pw))
        y0 = int(np.clip(py if py is not None else 0, 0, H - ph))
    elif mode == "random":
        x0 = int(np.random.randint(0, W - pw + 1))
        y0 = int(np.random.randint(0, H - ph + 1))
    else:
        raise ValueError(f"Unknown patch mode: {mode}")
    return (x0, y0, x0 + pw, y0 + ph)

def encode_img_to_latent(model, x):  # x: (B,3,h,w) in [-1,1]
    enc = model.encode_first_stage(x)
    z = model.get_first_stage_encoding(enc)
    return z

def make_out_path(root, rel_path):
    dst = os.path.join(root, rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    return dst

# ----------------- 批处理：一次处理 B 对 -----------------
def run_fusion_on_batch(model, device, ir_patches, vis_patches,
                        prompt, ddim_steps, ddim_eta, precision):
    """
    ir_patches/vis_patches: List[PIL.Image]，长度=B（同尺寸，如256x256）
    prompt: str（本次只用一个prompt；四个prompt分四次跑）
    返回：List[PIL.Image]（长度B），每张灰度L 256x256
    """
    assert len(ir_patches) == len(vis_patches) and len(ir_patches) > 0
    B = len(ir_patches)
    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope():
            with model.ema_scope():
                ir_x  = torch.cat([pil_to_tensor_nchw(im) for im in ir_patches], dim=0).to(device)
                vis_x = torch.cat([pil_to_tensor_nchw(im) for im in vis_patches], dim=0).to(device)

                z_ir  = encode_img_to_latent(model, ir_x)   # (B,3,h',w')
                z_vis = encode_img_to_latent(model, vis_x)  # (B,3,h',w')
                z_code = torch.cat([z_ir, z_vis], dim=1)    # (B,6,h',w')

                c = model.get_learned_conditioning([prompt] * B) if model.cond_stage_model is not None else None

                samples_z, _ = model.sample_log(
                    z_code=z_code, cond=c, batch_size=B,
                    ddim=True, ddim_steps=ddim_steps, eta=ddim_eta
                )
                x = model.decode_first_stage(samples_z)      # (B,C,h,w)
                x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

                outs = []
                if x.shape[1] == 1:
                    arr = (x[:,0].detach().cpu().numpy() * 255).astype(np.uint8)
                    for i in range(B):
                        outs.append(Image.fromarray(arr[i], mode="L"))
                else:
                    arr = (x[:,0].detach().cpu().numpy() * 255).astype(np.uint8)
                    for i in range(B):
                        outs.append(Image.fromarray(arr[i], mode="L"))
                return outs

# ----------------- 主流程 -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", action="store_true", help="use ddpm_fusion pipeline (IR+VIS)")
    parser.add_argument("--ir", type=str, help="path to single IR image")
    parser.add_argument("--vis", type=str, help="path to single VIS image")
    parser.add_argument("--ir_dir", type=str, help="folder of IR images (batch mode)")
    parser.add_argument("--vis_dir", type=str, help="folder of VIS images (batch mode)")
    parser.add_argument("--pair_by", type=str, default="stem", choices=["stem", "name"], help="配对策略")

    # 单个 patch 的宽高（默认 256×256）
    parser.add_argument("--H", type=int, default=256, help="patch height")
    parser.add_argument("--W", type=int, default=256, help="patch width")

    # 选择 patch 位置
    parser.add_argument("--patch_mode", type=str, default="random",
                        choices=["topleft", "center", "random", "xy"],
                        help="从公共区域选取 patch 的方式")
    parser.add_argument("--px", type=int, default=0, help="当 patch_mode=xy 时的左上角 x")
    parser.add_argument("--py", type=int, default=0, help="当 patch_mode=xy 时的左上角 y")

    # Prompt 列表（四条默认）
    parser.add_argument("--prompts", type=str, nargs="*", default=[
        "The fused image with full infrared information and no visible information",
        "The fused image with mostly infrared information and little visible information",
        "The fused image with little infrared information and mostly visible information",
        "The fused image with no infrared information and full visible information",
        "The fused image with standard infrared information and standard visible information",
    ])

    parser.add_argument("--batch_size", type=int, default=16, help="每轮同时处理的图像对数量")
    parser.add_argument("--outdir", type=str, default="./DPO_pair/")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--config", type=str, default="./configs/fusion_anything/ldm-txt2fusion.yaml")
    parser.add_argument("--ckpt", type=str, default="./models/ldm_fusion_model.ckpt")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--precision", type=str, choices=["full", "autocast"], default="autocast")
    parser.add_argument("--plms", action='store_true')
    parser.add_argument("--dpm_solver", action='store_true')

    opt = parser.parse_args()
    if not opt.fusion:
        raise SystemExit("本脚本仅处理 --fusion(IR+VIS) 模式。请加 --fusion。")

    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.dpm_solver:
        _ = DPMSolverSampler(model)
    elif opt.plms:
        _ = PLMSSampler(model)
    else:
        _ = DDIMSampler(model)

    # 输出目录结构
    out_root = opt.outdir
    subdirs = {
        "ir":        os.path.join(out_root, "ir_patch"),
        "vis":       os.path.join(out_root, "vis_patch"),
        "p0":        os.path.join(out_root, "fused_fullIR"),
        "p1":        os.path.join(out_root, "fused_mostIR"),
        "p2":        os.path.join(out_root, "fused_mostVIS"),
        "p3":        os.path.join(out_root, "fused_fullVIS"),
        "p4":        os.path.join(out_root, "fused_standard"),  # 可选
    }
    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)

    def key_fn(p):
        name = os.path.basename(p)
        return os.path.splitext(name)[0] if opt.pair_by == "stem" else name

    # ---------- 单张 ----------
    if opt.ir and opt.vis:
        ir_img  = load_pil(opt.ir)
        vis_img = load_pil(opt.vis)

        Wc, Hc = intersect_size(ir_img.size, vis_img.size)
        ir_overlap  = ir_img.crop((0, 0, Wc, Hc))
        vis_overlap = vis_img.crop((0, 0, Wc, Hc))
        box = choose_patch_box(Wc, Hc, opt.W, opt.H, mode=opt.patch_mode, px=opt.px, py=opt.py)
        ir_patch  = ir_overlap.crop(box)
        vis_patch = vis_overlap.crop(box)

        # 保存用于融合的 patch
        stem = os.path.splitext(os.path.basename(opt.ir))[0] + ".png"
        ir_patch.save(os.path.join(subdirs["ir"],  stem))
        vis_patch.save(os.path.join(subdirs["vis"], stem))

        # 逐 prompt 融合并保存
        for idx, prompt in enumerate(opt.prompts):
            fused_list = run_fusion_on_batch(model, device, [ir_patch], [vis_patch],
                                             prompt, opt.ddim_steps, opt.ddim_eta, opt.precision)
            out_key = ["p0","p1","p2","p3"][idx] if idx < 4 else f"p{idx}"
            fused_list[0].convert("RGB").save(os.path.join(subdirs[out_key], stem))
        print(f"[fusion] saved 256x256 patches & fused results under: {opt.outdir}")
        return

    # ---------- 批量（B=--batch_size） ----------
    assert opt.ir_dir and opt.vis_dir, "batch 模式需要同时提供 --ir_dir 与 --vis_dir"
    ir_paths  = list_images(opt.ir_dir)
    vis_paths = list_images(opt.vis_dir)
    vis_index = {key_fn(p): p for p in vis_paths}
    base_ir_root = os.path.abspath(opt.ir_dir)

    # 先做配对列表
    paired_paths = [(ip, vis_index[key_fn(ip)]) for ip in ir_paths if key_fn(ip) in vis_index]

    paired, skipped = 0, 0
    B = max(1, int(opt.batch_size))

    for start in tqdm(range(0, len(paired_paths), B), desc="Fusing (batched patches)"):
        batch_pairs = paired_paths[start:start+B]
        ir_batch, vis_batch, rel_paths = [], [], []

        # 组装当前 batch 的 patch
        try:
            for (ir_p, vis_p) in batch_pairs:
                ir_img  = load_pil(ir_p)
                vis_img = load_pil(vis_p)

                Wc, Hc = intersect_size(ir_img.size, vis_img.size)
                ir_overlap  = ir_img.crop((0, 0, Wc, Hc))
                vis_overlap = vis_img.crop((0, 0, Wc, Hc))
                box = choose_patch_box(Wc, Hc, opt.W, opt.H, mode=opt.patch_mode, px=opt.px, py=opt.py)

                ir_patch  = ir_overlap.crop(box)
                vis_patch = vis_overlap.crop(box)

                ir_batch.append(ir_patch)
                vis_batch.append(vis_patch)

                rel = os.path.relpath(ir_p, base_ir_root)
                rel_noext = os.path.splitext(rel)[0] + ".png"
                rel_paths.append(rel_noext)

            # 先保存 IR/VIS patch
            for rp, irp, vsp in zip(rel_paths, ir_batch, vis_batch):
                ir_save  = make_out_path(subdirs["ir"],  rp)
                vis_save = make_out_path(subdirs["vis"], rp)
                irp.save(ir_save)
                vsp.save(vis_save)

            # 逐 prompt 跑一遍 B 前向并保存
            out_keys = ["p0","p1","p2","p3"]
            for p_idx, prompt in enumerate(opt.prompts):
                fused_list = run_fusion_on_batch(
                    model, device, ir_batch, vis_batch,
                    prompt=prompt, ddim_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, precision=opt.precision
                )
                key = out_keys[p_idx] if p_idx < len(out_keys) else f"p{p_idx}"
                for rp, fused_img in zip(rel_paths, fused_list):
                    fused_save = make_out_path(subdirs[key], rp)
                    fused_img.convert("RGB").save(fused_save)

            paired += len(batch_pairs)

        except Exception as e:
            print(f"[WARN] batch starting at {start}: {e}")
            skipped += len(batch_pairs)
            continue

    print(f"[fusion] paired={paired}, skipped={skipped}")
    print(f"Done. Results at: {opt.outdir}")

if __name__ == "__main__":
    main()
