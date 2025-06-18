#!/usr/bin/env python
import os
import random
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from skimage import color
from tqdm.auto import tqdm

from lib.lime import LIME
from lib.fog_gen import fogAttenuation
from lib.rain_gen import RainGenUsingNoise
from lib.gen_utils import (illumination2opacity, 
                           layer_blend, 
                           alpha_blend, 
                           reduce_lightHSV, 
                           scale_depth)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear_path", type=str, required=True, help="path to the file or the folder")
    parser.add_argument("--depth_path", type=str, required=True, help="path to the file or the folder")
    parser.add_argument("--save_folder", type=str, default="./generated/", help="path to the folder")
    parser.add_argument("--txt_file", default=None, help="path to the folder")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="display detailed processing information")
    return parser.parse_args()

class RainEffectGenerator:
    def __init__(self):
        self._lime = LIME(iterations=25, alpha=1.0)
        # self._illumination2darkness = {0: 1, 1: 0.75, 2: 0.65, 3: 0.5}
        self._illumination2darkness = {0: 1, 1: 0.95, 2: 0.85, 3: 0.8}
        self._weather2visibility = (1000, 2000)
        # self._weather2visibility = {'fog': (100,250), 'rain': (1000,2000), 'snow': (500, 1000)}
        # self._illumination2fogcolor = {0: (80, 120), 1: (120, 160), 2: (160, 200), 3: (200, 240)}
        self._illumination2fogcolor = {0: (150, 180), 1: (180, 200), 2: (200, 240), 3: (200, 240)}
        self._rain_layer_gen = RainGenUsingNoise()
        
    def getIlluminationMap(self, img: np.ndarray) -> np.ndarray: 
        self._lime.load(img)
        T = self._lime.illumMap()
        return T
    
    def getIlluminationMapCheat(self, img: np.ndarray) -> np.ndarray: 
        T = color.rgb2gray(img)
        return T
          
    def genRainLayer(self, h=720, w=1280):
        blur_angle = random.choice([-1, 1])*random.randint(60, 90)
        layer_large = self._rain_layer_gen.genRainLayer(h=720, 
                                                        w=1280, 
                                                        noise_scale=random.uniform(0.35, 0.55), 
                                                        noise_amount=0.2, 
                                                        zoom_layer=random.uniform(1.0, 3.5),
                                                        blur_kernel_size=random.choice([15, 17, 19, 21, 23]), 
                                                        blur_angle=blur_angle
                                                        )#large
        
        layer_small = self._rain_layer_gen.genRainLayer(h=720, 
                                                        w=1280, 
                                                        noise_scale=random.uniform(0.35, 0.55), 
                                                        noise_amount=0.15, 
                                                        zoom_layer=random.uniform(1.0, 3.5),
                                                        blur_kernel_size=random.choice([7, 9, 11, 13]), 
                                                        blur_angle=blur_angle
                                                        )#small
        layer = layer_blend(layer_small, layer_large)
        hl, wl = layer.shape

        if h!=hl or w!=wl:
            layer = np.asarray(Image.fromarray(layer).resize((w, h)))
        return layer
    
    def genEffect(self, img_path: str, depth_path: str):
        I = np.array(Image.open(img_path))
        D = np.load(depth_path)
        
        hI, wI, _ = I.shape
        hD, wD = D.shape
        
        if hI!=hD or wI!=wD:
            D = scale_depth(D, hI, wI)
        
        T = self.getIlluminationMapCheat(I)
        illumination_array = np.histogram(T, bins=4, range=(0,1))[0]/(T.size)
        illumination = illumination_array.argmax()
        
        if illumination>0:
            visibility = visibility = random.randint(self._weather2visibility[0], self._weather2visibility[1])
            fog_color = random.randint(self._illumination2fogcolor[illumination][0], self._illumination2fogcolor[illumination][1])
            I_dark = reduce_lightHSV(I, sat_red=self._illumination2darkness[illumination], val_red=self._illumination2darkness[illumination])
            I_fog = fogAttenuation(I_dark, D, visibility=visibility, fog_color=fog_color)
        else:
            fog_color = 75
            visibility = D.max()*0.75 if D.max()<1000 else 750
            I_fog = fogAttenuation(I, D, visibility=visibility, fog_color=fog_color)
        
        alpha = illumination2opacity(I, illumination)*random.uniform(0.3, 0.5)
        rain_layer = self.genRainLayer(h=hI, w=wI) 
        I_rain = alpha_blend(I_fog, rain_layer, alpha)
        return I_rain.astype(np.uint8)

def main():
    args = parse_arguments()
    raingen = RainEffectGenerator()
    
    clearP = Path(args.clear_path)
    depthP = Path(args.depth_path)
    save_folder = Path(args.save_folder)

    if not save_folder.exists():
        os.makedirs(str(save_folder))
        print(f"Created output directory: {save_folder}")
    
    # Case 1: Single image and single depth file
    if clearP.is_file() and (depthP.is_file() and depthP.suffix==".npy"):
        rainy = raingen.genEffect(clearP, depthP)
        save_path = save_folder / (clearP.stem + "-rsyn.jpg")
        print(f"Saving rainy image to {save_path}")
        Image.fromarray(rainy).save(save_path)
        if args.show:
            Image.fromarray(rainy).show()
    
    # Case 2: Directory of images
    elif clearP.is_dir():
        # Get list of valid image files
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        if args.txt_file:
            with open(args.txt_file, 'r') as f:
                files = f.read().strip().split('\n')
            image_files = [clearP / f for f in files if f.strip()]
        else:
            image_files = [f for f in sorted(Path(clearP).glob("*")) 
                          if f.is_file() and f.suffix.lower() in valid_extensions]
        
        print(f"Found {len(image_files)} images to process")
        
        # Case 2a: Depth path is a directory
        if depthP.is_dir():
            for img_path in tqdm(image_files, desc="Processing images"):
                try:
                    # Try to find matching depth file
                    depth_file = depthP / f"{img_path.stem}.npy"
                    if not depth_file.exists():
                        # Alternative naming formats to try
                        alt_depth_files = list(depthP.glob(f"{img_path.stem}*.npy"))
                        if alt_depth_files:
                            depth_file = alt_depth_files[0]
                        else:
                            print(f"Warning: No depth file found for {img_path.name}, skipping.")
                            continue
                    
                    rainy = raingen.genEffect(str(img_path), str(depth_file))
                    save_path = save_folder / f"{img_path.stem}-rsyn.jpg"
                    Image.fromarray(rainy).save(save_path)
                    if args.verbose:
                        print(f"Processed {img_path.name} -> {save_path.name}")
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
        
        # Case 2b: Depth path is a single file (probably containing multiple depth maps)
        elif depthP.is_file() and depthP.suffix == ".npy":
            try:
                # Process each image with the same depth file
                for img_path in tqdm(image_files, desc="Processing images"):
                    try:
                        rainy = raingen.genEffect(str(img_path), str(depthP))
                        save_path = save_folder / f"{img_path.stem}-rsyn.jpg"
                        Image.fromarray(rainy).save(save_path)
                        if args.verbose:
                            print(f"Processed {img_path.name} -> {save_path.name}")
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
            except Exception as e:
                print(f"Error loading depth file: {str(e)}")
        else:
            print(f"Invalid depth path: {depthP}. It should be a .npy file or a directory.")
    else:
        print("Invalid combination of arguments. Please provide either:")
        print("1. A single clear image file and a single depth .npy file, or")
        print("2. A directory of clear images and either a directory of depth files or a single depth .npy file")

if __name__=='__main__':
    main()

