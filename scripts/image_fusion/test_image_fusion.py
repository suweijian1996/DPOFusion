import os
import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image, make_grid
from einops import rearrange
from ldm.util import instantiate_from_config
from torch.utils.data import DataLoader
from tqdm import tqdm

@torch.no_grad()
def main():
    config_path = "./configs/fusion_anything/ldm-llvip-fa-f4-test.yaml"
    # ckpt_path = "/path/to/model.ckpt"
    output_dir = "./outputs/temp"
    os.makedirs(output_dir, exist_ok=True)

    # Load config and model
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    # ckpt = torch.load(ckpt_path, map_location="cpu")
    # model.load_state_dict(ckpt["state_dict"], strict=False)
    model.cuda().eval()

    # Load val dataloader
    val_cfg = config.data.params.validation
    dataset = instantiate_from_config(val_cfg)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    ddim_steps = 50
    eta = 1.0

    for i, batch in enumerate(tqdm(val_loader, desc="Sampling from val set")):
        if i >= 10:
            break

        with model.ema_scope("Sampling"):
            z_ir, c, x_vis, x_vis_rec, _ = model.get_input(batch, "ir",
                                                           return_first_stage_outputs=True,
                                                           force_c_encode=True,
                                                           return_original_cond=True,
                                                           bs=1)
            z_vis, _, x_ir, x_ir_rec, _ = model.get_input(batch, "vis",
                                                          return_first_stage_outputs=True,
                                                          force_c_encode=True,
                                                          return_original_cond=True,
                                                          bs=1)

            # 🔧 CORRECT: do fusion like training
            z_code = torch.cat([z_ir, z_vis], dim=1)

            # 🔧 CORRECT: use model.sample_log instead of bare sampler
            samples, _ = model.sample_log(
                z_code=z_code,
                cond=c,
                batch_size=z_code.shape[0],
                ddim=True,
                ddim_steps=ddim_steps,
                eta=eta
            )

            x_samples = model.decode_first_stage(samples)

        # Save each sample
        for j in range(x_samples.shape[0]):
            save_image((x_samples[j] + 1) / 2.0, os.path.join(output_dir, f"fusion_sample_{i}_{j}.png"))
            save_image((x_ir[j] + 1) / 2.0, os.path.join(output_dir, f"ir_input_{i}_{j}.png"))
            save_image((x_vis[j] + 1) / 2.0, os.path.join(output_dir, f"vis_input_{i}_{j}.png"))

        # Save grid
        x_ir = x_ir[:, 0:1, :, :]
        x_vis = x_vis[:, 0:1, :, :]
        grid = make_grid(torch.cat([
            (x_ir + 1) / 2.0,
            (x_vis + 1) / 2.0,
            (x_samples + 1) / 2.0
        ], dim=0), nrow=x_ir.shape[0])
        save_image(grid, os.path.join(output_dir, f"grid_{i}.png"))

        print(f"Saved batch {i}")

if __name__ == "__main__":
    main()
