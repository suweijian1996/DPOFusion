# DPOFusion

This is the official repository for **DPOFusion**. 

## Environment Setup

Our project is built upon the environment of [Stable Diffusion (CompVis)](https://github.com/CompVis/stable-diffusion). Please follow the steps below to configure your environment:

1. Clone this repository:
   ```bash
   git clone https://github.com/suweijian1996/DPOFusion.git
   cd DPOFusion
   ```

2. Create and activate the conda environment following the LDM installation guide:
   ```bash
   conda env create -f environment.yaml
   conda activate ldm
   ```

3. Install additional dependencies if needed:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Demo

The **General Model** is capable of handling most tasks and is recommended for quick testing. Download the required model checkpoints and place them in the corresponding directories:

| Model | Download Link | Directory |
|-------|--------------|-----------|
| General Model | [Download](https://drive.google.com/file/d/12YuplfOZKxBSAScoa121qlcaljvQ-v1n/view?usp=drive_link) | `models/` |
| Fusion Layer  | [Download](https://drive.google.com/file/d/1FF1wqDx66laYHG1O2Rqv81Kw0D1WPZPi/view?usp=drive_link) | `models/fusion_layer_models/` |
| VAE Encoder | [Download](https://drive.google.com/file/d/1ucQg0E4IqojRAnpXZQwBN4f7TTDMeRU2/view?usp=drive_link) | `models/first_stage_models/kl_f4/` |

After setting up the environment and downloading the models, you can run a quick demo:

```bash
python control_fused.py ./data/infrared ./data/visible \
    --num_samples 1 \
    --ddim_steps 50 \
    --strength 1.0 \
    --scale 1.0 \
    --seed 23 \
    --eta 1.0 \
    --factor 1.0
```

**Note:** If you encounter out-of-memory issues, you can adjust the `--factor` parameter to scale down the input images (e.g., `--factor 2`).

For more models and advanced options, please visit our [Google Drive folder](https://drive.google.com/drive/folders/1-JBVAQ3Mc5w8Bkq-33VYVggcYn8V2m2b?usp=drive_link). To use a different model, modify the model path in `control_fused.py`:
```python
model.load_state_dict(load_state_dict('./models/General_model.ckpt', location='cuda'))
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@InProceedings{Su_2026_CVPR,
    author    = {Su, Weijian and Zhang, Songqian and Han, Yuqi and Zhuang, Jian and Huang, Yongdong and Zhang, Qiang},
    title     = {Fusion in Your Way: Aligning Image Fusion with Heterogeneous Demands via Direct Preference Optimization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {41499-41509}
}
```

## Acknowledgments

This project is built upon the excellent work of:
- [Latent Diffusion Models (LDM)](https://github.com/CompVis/stable-diffusion) by CompVis
- [ControlNet](https://github.com/lllyasviel/ControlNet) by Lvmin Zhang

We are grateful for their contributions to the open-source community.
