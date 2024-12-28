from PIL import Image
import numpy as np

import torch
import torch.nn
import torch.optim as optim
from torchvision import transforms, models

import clip
import torch.nn.functional as F
from clip_loss.template import imagenet_templates

from PIL import Image 
import PIL 
import argparse
from torchvision import utils as vutils
from torchvision.transforms.functional import adjust_contrast

def clip_normalize(image,device):
    image = F.interpolate(image,size=224,mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    
    return loss_var_l2

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]

def load_image2(img_path, img_height=None,img_width =None):
    
    image = Image.open(img_path)
    if img_width is not None:
        image = image.resize((img_width, img_height))  # change image size to (3, img_size, img_size)
    
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        ])   

    image = transform(image)[:3, :, :].unsqueeze(0)

    return image

def img_normalize(image,device):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image
def get_features(image, model, layers=None):

    if layers is None:
        layers = {'0': 'conv1_1',  
                  '5': 'conv2_1',  
                  '10': 'conv3_1', 
                  '19': 'conv4_1', 
                  '21': 'conv4_2', 
                  '28': 'conv5_1',
                  '31': 'conv5_2'
                 }  
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)   
        if name in layers:
            features[layers[name]] = x
    
    return features

#function
def get_clip_loss(content_image,target,prompt,device):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lambda_tv', type=float, default=2e-3,
                        help='total variation loss parameter')
    parser.add_argument('--lambda_patch', type=float, default=9000,
                        help='PatchCLIP loss parameter')
    parser.add_argument('--lambda_dir', type=float, default=500,
                        help='directional loss parameter')
    parser.add_argument('--lambda_c', type=float, default=150,
                        help='content loss parameter')
    #crop_size 224
    parser.add_argument('--crop_size', type=int, default=224,
                        help='cropped image size')
    parser.add_argument('--num_crops', type=int, default=64,
                        help='number of patches')
    parser.add_argument('--img_width', type=int, default=512,
                        help='size of images')
    parser.add_argument('--img_height', type=int, default=512,
                        help='size of images')
    parser.add_argument('--thresh', type=float, default=0.7,
                        help='Number of domains')
    args = parser.parse_args()
    
    VGG = models.vgg19(pretrained=True).features
    VGG.to(device)
    content_features = get_features(img_normalize(content_image,device), VGG)
    target_features = get_features(img_normalize(target,device), VGG)
    
    #content_loss = 0

    #content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    #content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)
    
    cropper = transforms.Compose([
        transforms.RandomCrop(args.crop_size)
    ])
    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
        transforms.Resize(224)
    ])
    
    clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)
    source = "a Photo"
    with torch.no_grad():
        template_text = compose_text_with_templates(prompt, imagenet_templates)
        tokens = clip.tokenize(template_text).to(device)
        text_features = clip_model.encode_text(tokens).detach()
        text_features = text_features.mean(axis=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        template_source = compose_text_with_templates(source, imagenet_templates)
        tokens_source = clip.tokenize(template_source).to(device)
        text_source = clip_model.encode_text(tokens_source).detach()
        text_source = text_source.mean(axis=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)
        source_features = clip_model.encode_image(clip_normalize(content_image,device))
        source_features /= (source_features.clone().norm(dim=-1, keepdim=True))
        
    loss_patch=0 
    img_proc =[]
    for n in range(args.num_crops):
        target_crop = cropper(target)
        target_crop = augment(target_crop)
        img_proc.append(target_crop)
    img_proc = torch.cat(img_proc,dim=0)
    img_aug = img_proc
    
    image_features = clip_model.encode_image(clip_normalize(img_aug,device))
    image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
    
    img_direction = (image_features-source_features)
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)
    
    text_direction = (text_features-text_source).repeat(image_features.size(0),1)
    text_direction /= text_direction.norm(dim=-1, keepdim=True)
    
    loss_temp = (1- torch.cosine_similarity(img_direction, text_direction, dim=1))
    loss_temp[loss_temp<args.thresh] =0
    loss_patch+=loss_temp.mean()
    
    glob_features = clip_model.encode_image(clip_normalize(target,device))
    glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))
    
    glob_direction = (glob_features-source_features)
    glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)
    
    loss_glob = (1- torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()
    
    reg_tv = args.lambda_tv*get_image_prior_losses(target)
    
    #total_loss = args.lambda_patch*loss_patch + args.lambda_c * content_loss+ reg_tv+ args.lambda_dir*loss_glob
    total_loss = args.lambda_dir*loss_glob+ args.lambda_patch*loss_patch
    return total_loss
