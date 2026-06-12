import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.fusion.NetUtils import RestormerFusionLayer

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2



class AutoFusionEncoder(pl.LightningModule):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 concat_mode=True,
                 lossconfig = None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 embed_dim = 3,
                 scheduler_config=None,
                 ckpt_path = None,
                 base_learning_rate = 1e-4):
        self.scale_by_std = scale_by_std
        # for backwards compatibility after implementation of DiffusionWrapper
        ignore_keys = []
        super().__init__()
        self.concat_mode = concat_mode
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.clip_denoised = False
        self.bbox_tokenizer = None  
        self.embed_dim = embed_dim
        self.fusion_layer = RestormerFusionLayer(in_channels=self.embed_dim*2,embed_channels=48, out_channels=self.embed_dim)
        self.scheduler_config = scheduler_config
        self.learning_rate = base_learning_rate
        self.loss = instantiate_from_config(lossconfig)

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True


    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    
    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x_ir = self.get_input_from_key(batch, "ir")
            x_vis = self.get_input_from_key(batch, "vis")
            x_ir = x_ir.to(self.device)
            x_vis = x_vis.to(self.device)
            z_ir = self.get_first_stage_encoding(self.encode_first_stage(x_ir)).detach()
            z_vis = self.get_first_stage_encoding(self.encode_first_stage(x_vis)).detach()
            z_all = torch.cat([z_ir, z_vis], dim=0)
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z_all.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
            

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_input_from_key(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, bs=None):
        x_ir = self.get_input_from_key(batch, "ir")
        x_vis = self.get_input_from_key(batch, "vis")

        if bs is not None:
            x_ir = x_ir[:bs]
            x_vis = x_vis[:bs]
        x_ir = x_ir.to(self.device)
        x_vis = x_vis.to(self.device)
        encoder_posterior_ir = self.encode_first_stage(x_ir)
        encoder_posterior_vis = self.encode_first_stage(x_vis)
        z_ir = self.get_first_stage_encoding(encoder_posterior_ir).detach()
        z_vis = self.get_first_stage_encoding(encoder_posterior_vis).detach()
        out = [z_ir, z_vis]
        if return_first_stage_outputs:
            z_fusion = self.fusion_layer(z_ir, z_vis)
            xrec = self.decode_first_stage(z_fusion)
            out.extend([x_ir, xrec])
        return out

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        x = self.first_stage_model.decode(z)
        x = x.mean(dim=1, keepdim=True)   # (B, 3, H, W) → (B, 1, H, W)
        return x


    # same as above but without decorator
    def differentiable_decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        x = self.first_stage_model.decode(z)
        x = x.mean(dim=1, keepdim=True)   # (B, 3, H, W) → (B, 1, H, W)
        return x
    
    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def forward(self, x, *args, **kwargs):
        # TODO calu loss
        x_ir = self.get_input_from_key(x, "ir")
        x_vis = self.get_input_from_key(x, "vis")
        x_ir = x_ir.to(self.device)
        x_vis = x_vis.to(self.device)
        encoder_posterior_ir = self.encode_first_stage(x_ir)
        encoder_posterior_vis = self.encode_first_stage(x_vis)
        z_ir = self.get_first_stage_encoding(encoder_posterior_ir)
        z_vis = self.get_first_stage_encoding(encoder_posterior_vis)
        z_fusion = self.fusion_layer(z_ir, z_vis)
        out = self.differentiable_decode_first_stage(z_fusion)
        # TODO 集成到modules

        # mse_loss = nn.MSELoss()
        # target = torch.maximum(x_ir, x_vis)  # 保证梯度正确传播
        loss = self.loss(out, x_vis, x_ir)
        return loss

    def training_step(self, batch, batch_idx):
        x_ir = self.get_input_from_key(batch, "ir")
        x_vis = self.get_input_from_key(batch, "vis")
        x_ir = x_ir.to(self.device)
        x_vis = x_vis.to(self.device)
        encoder_posterior_ir = self.encode_first_stage(x_ir)
        encoder_posterior_vis = self.encode_first_stage(x_vis)
        z_ir = self.get_first_stage_encoding(encoder_posterior_ir)
        z_vis = self.get_first_stage_encoding(encoder_posterior_vis)
            

        z_fusion = self.fusion_layer(z_ir, z_vis)
        out = self.differentiable_decode_first_stage(z_fusion)
        loss = self.loss(out, x_vis, x_ir)

        return loss

    
    def validation_step(self, batch, batch_idx):
        # 获取输入
        x_ir = self.get_input_from_key(batch, "ir").to(self.device)
        x_vis = self.get_input_from_key(batch, "vis").to(self.device)

        # 编码 → 融合 → 解码
        encoder_posterior_ir = self.encode_first_stage(x_ir)
        encoder_posterior_vis = self.encode_first_stage(x_vis)
        z_ir = self.get_first_stage_encoding(encoder_posterior_ir)
        z_vis = self.get_first_stage_encoding(encoder_posterior_vis)

        z_fusion = self.fusion_layer(z_ir, z_vis)
        with torch.no_grad():  # 避免显存爆炸
            out = self.decode_first_stage(z_fusion)

        # 融合监督目标：最大值（可换成 avg 或 x_ir）
        loss = self.loss(out, x_vis, x_ir)

        # 日志记录（可以在 TensorBoard 中看到）
        self.log("val/fusion_loss", loss,
                prog_bar=True, logger=True,
                on_step=False, on_epoch=True, sync_dist=True)

        return {"val_loss": loss}

    # @torch.no_grad()
    # def log_images(self, batch, only_inputs=False, **kwargs):
    #     log = dict()
    #     x = self.get_input(batch, self.image_key)
    #     x = x.to(self.device)
    #     if not only_inputs:
    #         xrec, posterior = self(x)
    #         if x.shape[1] > 3:
    #             # colorize with random projection
    #             assert xrec.shape[1] > 3
    #             x = self.to_rgb(x)
    #             xrec = self.to_rgb(xrec)
    #         log["samples"] = self.decode(torch.randn_like(posterior.sample()))
    #         log["reconstructions"] = xrec
    #     log["inputs"] = x
    #     return log
    
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()

        # 获取输入
        x_ir = self.get_input_from_key(batch, "ir").to(self.device)
        x_vis = self.get_input_from_key(batch, "vis").to(self.device)

        # 融合目标（比如 max 融合）
        target = torch.maximum(x_ir, x_vis)

        # 编码 → 融合 → 解码
        z_ir = self.get_first_stage_encoding(self.encode_first_stage(x_ir))
        z_vis = self.get_first_stage_encoding(self.encode_first_stage(x_vis))
        z_fusion = self.fusion_layer(z_ir, z_vis)
        xrec = self.decode_first_stage(z_fusion)

        # 图像色彩可视化处理（可选）
        # if x_ir.shape[1] > 3:
        #     x_ir = self.to_rgb(x_ir)
        #     x_vis = self.to_rgb(x_vis)
        #     target = self.to_rgb(target)
        #     xrec = self.to_rgb(xrec)

        # 组织 log
        log["ir"] = x_ir
        log["vis"] = x_vis
        log["target"] = target
        log["reconstructions"] = xrec

        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.fusion_layer.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x