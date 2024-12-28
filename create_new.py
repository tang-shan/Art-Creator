import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler,UNet2DConditionModel
import os
from tqdm import tqdm
from PIL import Image
import numpy as np


device=torch.device("cuda") 
unet = UNet2DConditionModel.from_pretrained("./checkpoints/unet")
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",use_auth_token=MY_TOKEN, unet=unet)
model = model.to(device)

prompt="a robot created by Wu Guanzhong"
for i in range(50):
    image = model(prompt).images[0]
    image.save('result.png')