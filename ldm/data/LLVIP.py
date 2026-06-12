# coding: utf-8
"""
LLVIP custom dataloaders for STABLE DIFFUSION

Folder structure (example) ▾
DATA_ROOT/
    infrared/
        train/
            scene0001.png
            scene0002.png
            ...
        val/
            ...
    visible/
        train/
            scene0001.png
            scene0002.png
        val/
            ...
    pref/
        train/
            scene0001.png
            ...
    mask/
        train/
            scene0001.png
            ...
    txt/
        train.txt          # comma‑separated:  <filename>,<prompt>          (prompt can be empty)
        val.txt

All images sharing the same *basename* (e.g. ``scene0001``) are treated as a pair.

Three sub‑datasets are provided:

* **LLVIPFusion[Train|Val]** ── returns only IR+VIS images.
* **LLVIPAugSR[Train|Val]** ── returns IR/VIS (+ LR/augmented variants) *plus* an auto‑generated english prompt.
* **LLVIPFinetune[Train|Val]** ── returns IR/VIS + pref + mask + user prompt (falls back to auto prompt).

All three inherit from **LLVIPBase** which mimics the ImageNetBase style: it builds a list of relative
file paths (``self.relpaths``) and holds a small dict with labels/metadata.  If ``process_images`` is
``True`` the images are actually loaded & pre‑processed to `[-1,1]`; otherwise only strings are
returned (useful for quick debugging).

External deps: albumentations, cv2, PIL (same as original).
"""

from __future__ import annotations

import os, json, random
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import cv2, albumentations
from PIL import Image
from omegaconf import OmegaConf
from functools import partial
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from taming.data.imagenet import retrieve

__all__ = [
    "LLVIPBase",
    "LLVIPFusionTrain", "LLVIPFusionVal",
    "LLVIPAugTrain", "LLVIPAugVal",
    "LLVIPFinetuneTrain", "LLVIPFinetuneVal",
]

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def _rgb(image: Image.Image) -> Image.Image:
    """Ensure PIL image in RGB mode."""
    return image.convert("RGB") if image.mode != "RGB" else image


def _pil_to_np(image: Image.Image) -> np.ndarray:
    return np.array(image).astype(np.uint8)


def _normalize(arr: np.ndarray) -> np.ndarray:
    return (arr / 127.5 - 1.0).astype(np.float32)


# --------------------------------------------------------------------------------------
# Base dataset
# --------------------------------------------------------------------------------------

class LLVIPBase(Dataset):
    """Base dataloader – iterates over scene basenames and fetches files from sub‑folders."""

    def __init__(self,
                 root: str,
                 split: str = "train",
                #  config: Any = None,
                 process_images: bool = True,
                 keep_text: bool = True,
                 size: int = 256,              # <-- 添加这行
                 random_crop: bool = False,     # <-- 以及这行（可选）
                 ):  # noqa: D401
        self.root = root
        self.split = split  # train / val / test
        # # self.config = OmegaConf.to_container(config or {}) if not isinstance(config, dict) else config
        # self.config = config if isinstance(config, dict) else {}
        self.process_images = process_images
        self.keep_text = keep_text

        # basic folders
        self.dir_ir = os.path.join(root, "infrared", split)
        self.dir_vis = os.path.join(root, "visible", split)
        self.dir_pref = os.path.join(root, "pref", split)  # may not exist for all splits
        self.dir_mask = os.path.join(root, "mask", split)

        # discover basenames (file stem w/o extension)
        self.basenames = self._scan_basenames()
        self.text_map = self._load_text_file()

        # imagenet‑style label dict (can be extended)
        self.labels: Dict[str, np.ndarray] = {"basename": np.array(self.basenames)}

        self.size = size
        self.random_crop = random_crop

    # ------------------------------------------------------------------
    # mandatory torch.utils.data.Dataset API
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.basenames)

    @staticmethod
    def _try_ext(folder: str, default=".png") -> str:
        """Return first file-extension inside folder, or <default> if folder缺失/为空."""
        if not os.path.isdir(folder):
            return default
        for f in os.listdir(folder):
            return os.path.splitext(f)[1]      # 找到第一个文件就返回
        return default                         # 文件夹空，返回默认


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.basenames[idx]
        example: Dict[str, Any] = {"basename": base}

        # ------------------------- 组装必要路径 -------------------------
        ir_path  = os.path.join(self.dir_ir,  base + self._ext(self.dir_ir))
        vis_path = os.path.join(self.dir_vis, base + self._ext(self.dir_vis))
        paths = {"ir": ir_path, "vis": vis_path}

        # -------------------- 仅当文件夹存在时才去尝试 pref/mask --------------------
        if os.path.isdir(self.dir_pref):
            paths["pref"] = os.path.join(self.dir_pref, base + self._try_ext(self.dir_pref))
        if os.path.isdir(self.dir_mask):
            paths["mask"] = os.path.join(self.dir_mask, base + self._try_ext(self.dir_mask))

        # 把真实存在的文件放进 example
        for k, p in paths.items():
            if os.path.exists(p):
                example[f"file_{k}"] = p

        # ------------------------- 加载/预处理 -------------------------
        if self.process_images:
            # 加载 IR 图像作为尺寸参考
            ir_img = Image.open(paths["ir"]).convert("L")
            ir_np = np.array(ir_img)
            h, w = ir_np.shape[:2]
            crop_size = self.size

            if self.random_crop:
                start_h = random.randint(0, h - crop_size)
                start_w = random.randint(0, w - crop_size)
            else:
                start_h = (h - crop_size) // 2
                start_w = (w - crop_size) // 2

            crop_coords = (start_h, start_w, crop_size)

            # 应用统一 crop
            example["ir"] = self._process_one(paths["ir"], crop_coords)
            example["vis"] = self._process_one(paths["vis"], crop_coords)
            if "pref" in example:
                example["pref"] = self._process_one(paths["pref"], crop_coords)
            if "mask" in example:
                m_img = Image.open(paths["mask"]).convert("L")
                m_np = np.array(m_img).astype(np.uint8)
                m_np = m_np[start_h:start_h + crop_size, start_w:start_w + crop_size]
                # m_np = cv2.resize(m_np, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                example["mask"] = _normalize(m_np)

        # ------------------------- 文本 -------------------------
        prompt = self.text_map.get(base)
        if self.keep_text and prompt is None:
            prompt = "The fused image"
        example["txt"] = prompt or ""

        # ------------------------- Finetune 强约束 -------------------------
        if isinstance(self, LLVIPFinetuneBase):
            assert "pref" in example and "mask" in example, \
                f"finetune sample {base} 缺少 pref 或 mask 文件"

        return example

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _scan_basenames(self) -> List[str]:
        """Collect basenames existing in both IR & VIS folder."""
        ir_files = set(os.path.splitext(f)[0] for f in os.listdir(self.dir_ir))
        vis_files = set(os.path.splitext(f)[0] for f in os.listdir(self.dir_vis))
        basenames = sorted(ir_files.intersection(vis_files))
        assert basenames, f"No paired images found in {self.dir_ir} and {self.dir_vis}"
        return basenames

    @staticmethod
    def _ext(folder: str) -> str:
        """Return first extension seen in folder (.png / .jpg)."""
        for f in os.listdir(folder):
            return os.path.splitext(f)[1]
        return ".png"

    def _process_one(self, path: str, crop_coords: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        img = Image.open(path).convert("L")
        img_np = np.array(img).astype(np.uint8)
        img_np = np.stack([img_np] * 3, axis=-1)

        # 统一裁剪位置
        if crop_coords is None:
            min_side = min(img_np.shape[:2])
            h, w = img_np.shape[:2]
            start_h = (h - min_side) // 2
            start_w = (w - min_side) // 2
            crop_size = min_side
        else:
            start_h, start_w, crop_size = crop_coords

        img_np = img_np[start_h:start_h + crop_size, start_w:start_w + crop_size]
        img_np = cv2.resize(img_np, (self.size, self.size), interpolation=cv2.INTER_AREA)

        return _normalize(img_np)

    # ------------------------------------------------------------------
    # text helpers
    # ------------------------------------------------------------------
    def _load_text_file(self) -> Dict[str, str]:
        txt_path = os.path.join(self.root, "txt", f"{self.split}.txt")
        mapping: Dict[str, str] = {}
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f.read().splitlines():
                    name, prompt = line.split(",", 1)
                    mapping[name] = prompt.strip()
        return mapping

# --------------------------------------------------------------------------------------
# 1) Simple Fusion dataset – only IR & VIS
# --------------------------------------------------------------------------------------

class LLVIPFusionTrain(LLVIPBase):
    def __init__(self, root: str, **kwargs):
        super().__init__(root, **kwargs)

class LLVIPFusionVal(LLVIPBase):
    def __init__(self, root: str, **kwargs):
        super().__init__(root, **kwargs)

# --------------------------------------------------------------------------------------
# 2) Augmented dataset for LDM pre‑training – includes auto prompt
# --------------------------------------------------------------------------------------

class LLVIPAugBase(LLVIPBase):
    def __init__(self, *args, downscale_f: int = 4, **kwargs):
        self.downscale_f = downscale_f
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        ex = super().__getitem__(idx)

        # 默认字段
        ex["aug"] = None
        ex["aug_src"] = None

        # augmentation logic
        mode = random.choice(["vis_texture", "ir_contrast"])
        if mode == "vis_texture":
            vis = ex["vis"].copy()
            k = random.choice(["blur", "sharpen"])
            if k == "blur":
                vis_lr = cv2.GaussianBlur(((vis + 1) * 127.5).astype(np.uint8), (5, 5), 0)
                prompt = "texture detail reduced"
            else:
                vis_lr = cv2.detailEnhance(((vis + 1) * 127.5).astype(np.uint8), sigma_s=10, sigma_r=0.15)
                prompt = "texture detail enhanced"

            # ex["aug"] = _normalize(vis_lr)
            ex["aug"] = vis
            ex["aug_src"] = "vis"

        else:
            ir = ex["ir"].copy()
            k = random.choice(["low", "high"])
            ir_img = ((ir + 1) * 127.5).astype(np.uint8)
            if k == "low":
                ir_lr = cv2.convertScaleAbs(ir_img, alpha=0.5, beta=0)
                prompt = "contrast decreased"
            else:
                ir_lr = cv2.convertScaleAbs(ir_img, alpha=1.5, beta=0)
                prompt = "contrast increased"

            # ex["aug"] = _normalize(ir_lr)
            ex["aug"] = ir
            ex["aug_src"] = "ir"

        ex["txt"] = prompt
        return ex

class LLVIPAugTrain(LLVIPAugBase):
    def __init__(self, root: str, **kwargs):
        super().__init__(root, **kwargs)

class LLVIPAugVal(LLVIPAugBase):
    def __init__(self, root: str, **kwargs):
        super().__init__(root, **kwargs)

# --------------------------------------------------------------------------------------
# 3) Finetune dataset – uses pref & mask & text files
# --------------------------------------------------------------------------------------

class LLVIPFinetuneBase(LLVIPBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        ex = super().__getitem__(idx)
        # require pref & mask
        assert "pref" in ex and "mask" in ex, "pref or mask missing for finetune"
        return ex

class LLVIPFinetuneTrain(LLVIPFinetuneBase):
    def __init__(self, root: str, **kwargs):
        super().__init__(root, split="train", **kwargs)

class LLVIPFinetuneVal(LLVIPFinetuneBase):
    def __init__(self, root: str, **kwargs):
        super().__init__(root, split="val", **kwargs)


# -------------------------------------------------------------------------------
# Quick test script
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    import imageio as imageio

    def to_uint8(arr: np.ndarray) -> np.ndarray:
        """[-1,1] -> uint8 RGB/BW"""
        arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        if arr.ndim == 2:                     # mask / grey
            return arr
        if arr.shape[0] in (1, 3):            # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        return arr

    parser = argparse.ArgumentParser(description="LLVIP dataset sanity-check dumper")
    parser.add_argument("--root", required=True,
                        help="LLVIP dataset root folder")
    parser.add_argument("--task", default="fusion",
                        choices=["fusion", "aug", "finetune"],
                        help="which dataset branch to test")
    parser.add_argument("--split", default="train",
                        choices=["train", "val"], help="subset")
    parser.add_argument("--save_dir", default="./llvip_test_dump",
                        help="where to dump images")
    parser.add_argument("--max_samples", type=int, default=20,
                        help="stop after N samples (0 = all)")
    parser.add_argument("--size", type=int, default=256,
                        help="resize target (matches training)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # choose dataset class
    cls_map = {
        ("fusion",  "train"): LLVIPFusionTrain,
        ("fusion",  "val"):   LLVIPFusionVal,
        ("aug",     "train"): LLVIPAugTrain,
        ("aug",     "val"):   LLVIPAugVal,
        ("finetune","train"): LLVIPFinetuneTrain,
        ("finetune","val"):   LLVIPFinetuneVal,
    }
    dset_cls = cls_map[(args.task, args.split)]
    dset = dset_cls(root=args.root, size=args.size, process_images=True)

    print(f"[INFO] Created {dset_cls.__name__}  |  size = {len(dset)}")

    n_out = args.max_samples if args.max_samples > 0 else len(dset)
    for i, sample in enumerate(tqdm(dset, total=n_out, desc="Dump")):
        if i >= n_out:
            break
        base = sample["basename"] if "basename" in sample else f"sample{i:04d}"
        subdir = os.path.join(args.save_dir, base)
        os.makedirs(subdir, exist_ok=True)

        # iterate keys that hold images
        for key in ["ir", "vis", "pref", "mask", "ir_aug", "vis_aug"]:
            if key in sample:
                img = sample[key]
                print(img.shape)
                img = to_uint8(img)
                print(img.shape)
                # HWC order expected by imageio
                if img.ndim == 3 and img.shape[2] == 1:
                    img = img[:, :, 0]
                imageio.imwrite(os.path.join(subdir, f"{key}.png"), img)

        # save prompt
        if "txt" in sample:
            with open(os.path.join(subdir, "prompt.txt"), "w") as fp:
                fp.write(sample["txt"] + "\n")

    print(f"[DONE] {n_out} samples written to {args.save_dir}")