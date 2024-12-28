import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler,UNet2DConditionModel
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from clip_loss.cliploss import get_clip_loss,load_image2
import torchvision

import time

def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image
 
def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents

def get_clip_grad(content_image,latent,timestep,prompt,device,model,noise_pred,context1,context2,alpha=7.5,beta=7.5):
    with torch.enable_grad():
        content_image = content_image.to(device)
        latent = latent.detach().requires_grad_()
        latents_input = torch.cat([latent] * 2)
        noise_pred_1 = model.unet(latents_input, timestep, encoder_hidden_states=context1)["sample"]
        noise_pred_uncond_1, noise_prediction_text1 = noise_pred_1.chunk(2)
        noise_pred = noise_pred_uncond_1 + alpha * (noise_prediction_text1 - noise_pred_uncond_1)
        
        alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
        z_0=(latent-(1-alpha_prod_t)**0.5*noise_pred)/(alpha_prod_t**0.5)
        latents = 1 / 0.18215 * z_0
        x_0 = model.vae.decode(latents)['sample']
        x_0 = (x_0+1)/2
        loss = get_clip_loss(content_image,x_0,prompt,device)
        grad = -torch.autograd.grad(loss, latent)[0]
    return grad

device=torch.device("cuda")    
batch_size = 1
height = width = 512
num_inference_steps = 50
start_time = 50
max_num_words = 77

generator=None

def style_transfer(latent,stage='texture',prompt1="a house at a lake",prompt2="picasso, abstract",unet=None,scheduler=None,model=None,my_save_path=None):
    if stage=='content':
        alpha=7.5
        beta=0
        lamda=0.0
        save_path='content.png'
    elif stage=='warp':
        alpha=7.5
        beta=7.5
        lamda=0.0
        save_path='warp.png'
        load_path='content.png'
        content_image=load_image2(img_path=load_path, img_height=height,img_width=width)
    else:
        alpha=7.5
        beta=7.5
        lamda=0.05
        threshold=0.5
        save_path='out.png'
        load_path='warp.png'
        content_image=load_image2(img_path=load_path, img_height=height,img_width=width)
    if my_save_path is not None:
        save_path=my_save_path
    text_input1 = model.tokenizer(
            prompt1,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
    text_input2 = model.tokenizer(
            prompt2,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
    text_embeddings1 = model.text_encoder(text_input1.input_ids.to(model.device))[0]
    text_embeddings2 = model.text_encoder(text_input2.input_ids.to(model.device))[0]
    _uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=model.tokenizer.model_max_length, return_tensors="pt"
        )
    _uncond_embeddings = model.text_encoder(_uncond_input.input_ids.to(model.device))[0]
    
    max_length = text_input1.input_ids.shape[-1]
    model.scheduler.set_timesteps(num_inference_steps)
    
    latent, latents = init_latent(x_t, model, height, width, generator, batch_size)
    vks=[]
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        
        with torch.no_grad():
            latents_input = torch.cat([latents] * 2)
            context1 = [_uncond_embeddings, text_embeddings1]
            context1 = torch.cat(context1)
    
            context2 = [_uncond_embeddings, text_embeddings2]
            context2 = torch.cat(context2)
    
            noise_pred_1 = model.unet(latents_input, t, encoder_hidden_states=context1)["sample"]
            noise_pred_2 = model.unet(latents_input, t, encoder_hidden_states=context2)["sample"]
            noise_pred_uncond_1, noise_prediction_text1 = noise_pred_1.chunk(2)
            noise_pred_uncond_2,noise_prediction_text2=noise_pred_2.chunk(2)
            u1=(1/scheduler.alphas[t]**0.5)*latents-(1-scheduler.alphas[t])/(scheduler.alphas[t]**0.5*(1-scheduler.alphas_cumprod[t])**0.5)*noise_prediction_text1
            u2=(1/scheduler.alphas[t]**0.5)*latents-(1-scheduler.alphas[t])/(scheduler.alphas[t]**0.5*(1-scheduler.alphas_cumprod[t])**0.5)*noise_prediction_text2
            if i<0:
                noise_pred = noise_pred_uncond_1 + alpha * (noise_prediction_text1 - noise_pred_uncond_1)
            else:
                K=u2/(u1+u2)
                noise_pred = noise_pred_uncond_1 + alpha * (noise_prediction_text1 - noise_pred_uncond_1)+beta * K * (noise_prediction_text2 - noise_pred_uncond_2)
            if lamda>0 and i>=100:
                clip_grad=get_clip_grad(content_image,latents.detach(),t,prompt2,device,model,noise_pred,context1,context2)
                latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
                diff=clip_grad-noise_pred
                diff=torch.linalg.norm(diff,dim=1)
                diff=(diff-diff.min())/(diff.max()-diff.min())
                diff=diff.unsqueeze(1)
                diff=diff.expand(-1,4,-1,-1)
                clip_grad[diff>threshold]=0            
                latents+= lamda*clip_grad
            else:
                latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    image=latent2image(model.vae, latents)[0]
    image=Image.fromarray(image)
    image.save(save_path)
    
unet = UNet2DConditionModel.from_pretrained("./checkpoints/unet")
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",use_auth_token=MY_TOKEN, scheduler=scheduler)
model = model.to(device)
tokenizer = model.tokenizer

x_t = torch.randn((1,4,64,64))
prompt="a woman"
style_transfer(latent=x_t,stage='texture', prompt1=prompt, prompt2="ink and wash painting",scheduler=scheduler,model=model)