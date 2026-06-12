from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

import time
from pytorch_lightning.callbacks import Callback



# Configs
resume_path = './models/control_model_wo_guide.ckpt'
batch_size = 8
logger_freq = 1000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
gpus = [0,1,2]


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./configs/fusion_anything/controlNet-llvip.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset(task='fusion')
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=gpus, precision=32, callbacks=[logger], accelerator='ddp', max_epochs=20, limit_val_batches=0)


# Train!
trainer.fit(model, dataloader)
