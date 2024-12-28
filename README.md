# Art Creator
![image](https://github.com/tang-shan/Art-Creator/blob/main/images/results.png)
The code for Art Creator: Steering Styles in Diffusion Model
## Setup
This code was tested with Python 3.11, Pytorch 2.1 using pre-trained models through huggingface / diffusers. Specifically, we implemented our method over Stable Diffusion 5.1. Additional required packages are listed in the requirements file. The code was tested on a Tesla V100 16GB but should work on other cards with at least 12GB VRAM.
## Pretrained Model
Download our pretrained model on ChinArt and WikiArt dataset on Baidu Disk:
链接: https://pan.baidu.com/s/1y3JBswAgly2GTmBQappX1g?pwd=4gp9 提取码: 4gp9 
and put the pretrained unet model to ./checkpoints folder.
## Usage
Generate an new artwork with our pretrained model: run create_new.py

Stylize an image with a text prompt: run text_control.py
