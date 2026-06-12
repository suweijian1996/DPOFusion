import torch
import cv2
import numpy as np
import einops
import random
from pytorch_lightning import seed_everything
import os
import argparse
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from annotator.util import resize_image, HWC3
import config



model = create_model('./configs/fusion_anything/controlNet-llvip-test.yaml').cpu()
model.load_state_dict(load_state_dict('./models/General_model.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(ir_image, vis_image, prompt, sd_prompt, num_samples, ddim_steps, strength, scale, seed, eta, factor):
    with torch.no_grad():
        ir_image = HWC3(ir_image)
        vis_image = HWC3(vis_image)

        H_orig, W_orig, C = ir_image.shape

        if factor >= 1:
            H_new = int(H_orig / factor) // 32 * 32
            W_new = int(W_orig / factor) // 32 * 32
        else:
            H_new = H_orig // 32 * 32
            W_new = W_orig // 32 * 32
        
        H_new = max(32, H_new)
        W_new = max(32, W_new)

        ir_resized = cv2.resize(ir_image, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
        vis_resized = cv2.resize(vis_image, (W_new, H_new), interpolation=cv2.INTER_LINEAR)

        ir_resized = torch.from_numpy(ir_resized).float().cuda() / 127.5 - 1.0
        vis_resized = torch.from_numpy(vis_resized).float().cuda() / 127.5 - 1.0

        ir_resized = torch.stack([ir_resized for _ in range(num_samples)], dim=0)
        vis_resized = torch.stack([vis_resized for _ in range(num_samples)], dim=0)

        ir_resized = einops.rearrange(ir_resized, 'b h w c -> b c h w').clone()
        vis_resized = einops.rearrange(vis_resized, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        cond = {"c_concat": [ir_resized], "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)], "sd_txt": [model.get_learned_conditioning([sd_prompt] * num_samples)]}
        shape = (3, H_new // 4, W_new // 4)

        model.control_scales = [strength] * 13
        z_ir_posterior = model.encode_first_stage(ir_resized)
        z_ir = model.get_first_stage_encoding(z_ir_posterior).detach()

        z_vis_posterior = model.encode_first_stage(vis_resized)
        z_vis = model.get_first_stage_encoding(z_vis_posterior).detach()

        z_code = torch.cat([z_ir, z_vis], dim=1)
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, z_code, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        # results = [x_samples[i] for i in range(num_samples)]
        results = []
        for i in range(num_samples):
            restored_img = cv2.resize(x_samples[i], (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            results.append(restored_img)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Image Diffusion with ControlNet (IR and VIS)")
    parser.add_argument("ir_folder", type=str, help="Path to the infrared images folder (IR)")
    parser.add_argument("vis_folder", type=str, help="Path to the visible images folder (VIS)")
    parser.add_argument("--prompt", type=str, default="Use standard mode to fuse images", help="Prompt for generating fused images")
    parser.add_argument("--sd_prompt", type=str, default="The fused image with standard infrared information and standard visible information", help="Additional Stable Diffusion prompt")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of images to generate (1-12)")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM steps (1-100)")
    parser.add_argument("--strength", type=float, default=1.0, help="Control strength (0.0-2.0)")
    parser.add_argument("--scale", type=float, default=1.0, help="Guidance scale (0.1-30.0)")
    parser.add_argument("--seed", type=int, default=23, help="Random seed (-1 for random)")
    parser.add_argument("--eta", type=float, default=1.0, help="DDIM eta value")
    parser.add_argument("--factor", type=float, default=1.0, help="When OOM occurs, reduce the input size.")
    return parser.parse_args()


def main():

    args = parse_args()


    if not os.path.exists(args.ir_folder) or not os.path.exists(args.vis_folder):
        print("The number of images in the infrared images folder and the visible light images folder does not match. Please check the folder contents.")
        return
    

    ir_files = sorted(os.listdir(args.ir_folder))
    vis_files = sorted(os.listdir(args.vis_folder))

    if len(ir_files) != len(vis_files):
        print("The number of images in the infrared images folder and the visible light images folder does not match. Please check the folder contents.")
        return

    for ir_filename, vis_filename in zip(ir_files, vis_files):

        ir_image_path = os.path.join(args.ir_folder, ir_filename)
        vis_image_path = os.path.join(args.vis_folder, vis_filename)

        ir_image = cv2.imread(ir_image_path)
        vis_image = cv2.imread(vis_image_path)

        if ir_image is None or vis_image is None:
            print(f"Unable to read image {ir_filename} or {vis_filename}, skipping this image pair.")
            continue


        results = process(ir_image, vis_image, args.prompt, args.sd_prompt, args.num_samples, args.ddim_steps, args.strength, args.scale, args.seed, args.eta, args.factor)


        mkdir_path = "./outputs/test_git"
        os.makedirs(mkdir_path, exist_ok=True)
        for i, result in enumerate(results):
            output_path = f"{mkdir_path}/{ir_filename}"
            cv2.imwrite(output_path, result)
            print(f"Fused Image Save as: {output_path}")

if __name__ == "__main__":
    main()
