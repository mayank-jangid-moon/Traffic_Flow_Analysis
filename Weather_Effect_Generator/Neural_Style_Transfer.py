import os
import copy
import torch
import random
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from lib.style_transfer_utils import (tensor2pil, 
                                      load_style_transfer_model, 
                                      run_style_transfer,
                                      style_content_image_loader)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-imgs", type=str, help="Path to the content images.", required=True)
    parser.add_argument("--style-imgs", type=str, help="Path to the style images.", required=True)
    parser.add_argument("--save-folder", type=str, help="Path to the save the generated images.", required=True)
    parser.add_argument("--vgg", type=str, help="Path to the pretrained VGG model.", required=True)

    parser.add_argument('--cuda', action='store_true', help="use cuda.")
    parser.add_argument('--ext', type=str, default="stl", help="extension for generated image.")
    parser.add_argument('--min-step', type=int, default=100, help="minimum iteration steps")
    parser.add_argument('--max-step', type=int, default=200, help="maximum iteration steps")
    parser.add_argument('--style-weight', type=float, default=100000, help="weight for style loss")
    parser.add_argument('--content-weight', type=float, default=2, help="weight for content loss")

    return parser.parse_args()


def transfer_style(cnn_path, 
                   cimg, 
                   simg,
                   min_step=100, 
                   max_step=200,
                   style_weight=100000,
                   content_weight=2,
                   device="cpu"
                  ):
    cnn = load_style_transfer_model(pretrained=cnn_path)
    
    content_img, style_img = style_content_image_loader(cimg, simg)
    input_img = copy.deepcopy(content_img).to(device, torch.float)

    output = run_style_transfer(cnn, 
                                content_img,
                                style_img,
                                input_img,  
                                num_steps=random.randint(min_step, max_step),
                                style_weight=style_weight, 
                                content_weight=content_weight,
                                device=device)
    return tensor2pil(output[0].detach().cpu())


def is_image_file(file_path):
    """Check if the file is an image based on its extension."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    return file_path.suffix.lower() in image_extensions

def main():
    args = parse_arguments()
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    content_path = Path(args.content_imgs)
    
    # Check if content_imgs is a directory, a single image file, or a text file with image paths
    if content_path.is_dir():
        # If it's a directory, get all image files from it
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        content_images = []
        for ext in image_extensions:
            content_images.extend(list(content_path.glob(f"*{ext}")))
            content_images.extend(list(content_path.glob(f"*{ext.upper()}")))
    elif is_image_file(content_path):
        # If it's a single image file
        content_images = [content_path]
    else:
        # If it's a file with a list of image paths
        try:
            with open(content_path, "r") as f:
                lines = f.read()
            content_images = [Path(line.strip()) for line in lines.split('\n') if line.strip()]
            # If paths are relative, make them absolute based on the content file's directory
            if not all(img.is_absolute() for img in content_images):
                base_dir = content_path.parent
                content_images = [base_dir / img if not img.is_absolute() else img for img in content_images]
        except UnicodeDecodeError:
            # If we can't read it as a text file, try as a single image
            content_images = [content_path]
    
    # Handle style images similar to content images
    style_path = Path(args.style_imgs)
    if style_path.is_dir():
        # If it's a directory, get all image files from it
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        style_images = []
        for ext in image_extensions:
            style_images.extend(list(style_path.glob(f"*{ext}")))
            style_images.extend(list(style_path.glob(f"*{ext.upper()}")))
    elif is_image_file(style_path):
        # If it's a single image file
        style_images = [style_path]
    else:
        raise ValueError(f"Style path {style_path} is not a valid directory or image file")

    save_folder = Path(args.save_folder)
    if not os.path.exists(args.save_folder):
        print(f"Creating {args.save_folder}")
        os.makedirs(str(save_folder))

    for cimg in tqdm(content_images):
        name = cimg.stem  # Gets the filename without extension
        extension = cimg.suffix[1:]  # Gets the extension without the dot
        
        # Use random.choice only if there are multiple style images, otherwise use the single one
        simg = random.choice(style_images) if len(style_images) > 1 else style_images[0]

        output_img = transfer_style(cnn_path=args.vgg, 
                                    cimg=cimg, 
                                    simg=simg, 
                                    min_step=args.min_step, 
                                    max_step=args.max_step,
                                    style_weight=args.style_weight, 
                                    content_weight=args.content_weight,
                                    device=device)
        output_img.save(save_folder / f"{name}-{args.ext}.{extension}")
        
if __name__ == '__main__':
    main()
