#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# This code accompanies the research paper: Upchurch, Paul, and Ransen
# Niu. "A Dense Material Segmentation Dataset for Indoor and Outdoor
# Scene Parsing." ECCV 2022.
#
# This example shows how to predict materials.
#

import argparse
from ast import If
from symbol import parameters
import torchvision.transforms as TTR
import os
import glob
import random
import json
import cv2
import numpy as np
import torch
import math
from PIL import Image

random.seed(112)

dms46 = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23,
    24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44, 46, 47, 48, 49,
    50, 51, 52, 53, 56, ]
t = json.load(open(os.path.expanduser('./taxonomy.json'), 'rb'))
srgb_colormap = [
    t['srgb_colormap'][i] for i in range(len(t['srgb_colormap'])) if i in dms46
]
srgb_colormap = np.array(srgb_colormap, dtype=np.uint8)

parameters = {"value_scale":255}
parameters["mean"] = [item * parameters["value_scale"] for item in [0.485, 0.456, 0.406]]
parameters["std"] = [item * parameters["value_scale"] for item in [0.229, 0.224, 0.225]]

images_to_infere = {}
def apply_color(label_mask):
    # translate labels to visualization colors
    vis = np.take(srgb_colormap, label_mask, axis=0)
    return vis[..., ::-1]

def apply_color2(label_mask):
    # translate labels to visualization colors
    # Begin Gil's code #
    material_dict = {}
    sum_of_pixels = 0
    # End Gil's code 
    vis = np.take(srgb_colormap, label_mask, axis=0)
    # Begin Gil's code
    for i in vis:
        for k in i:
            sum_of_pixels += 1
            for j in range(len(t['srgb_colormap'])):
                if np.array_equal(k,t['srgb_colormap'][j]):
                    if not t['names'][j] in material_dict:
                        material_dict[t['names'][j]] = 0
                    material_dict[t['names'][j]] +=1
                    break
    print(f"dict1: {material_dict}")
    for material in material_dict.keys():
        material_dict[material] /= sum_of_pixels
        material_dict[material] *= 100
    print(f"dict2: {material_dict}")
    # End Gil's code
    return vis[..., ::-1]


def fetch_prediciton(image_path, predicted_color):
    vis = predicted_color[..., ::-1]
    material_mapping = []
    for row in vis:
        material_mapping.append([])
        for rgb_array in row:
            images_to_infere[image_path]["num_of_pixels"] += 1
            for material_index in range(len(t['srgb_colormap'])):
                if np.array_equal(rgb_array,t['srgb_colormap'][material_index]):
                    if not material_index in images_to_infere[image_path]["materials"]:
                        images_to_infere[image_path]["materials"][material_index] = {"name":t['names'][material_index],
                                                                                     "rgb_color":t['srgb_colormap'][material_index], "num_of_pixels":0}
                    images_to_infere[image_path]["materials"][material_index]["num_of_pixels"] += 1
                    material_mapping[-1].append(t['names'][material_index])
                    break
    return material_mapping


        


def main(args):
    is_cuda = torch.cuda.is_available()
    model = torch.jit.load(args.jit_path)
    if is_cuda:
        model = model.cuda()

    images_list = glob.glob(f'{args.image_folder}/*')
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    os.makedirs(args.output_folder, exist_ok=True)

    for image_path in images_list:
        print(image_path)
        images_to_infere[image_path] = {"num_of_pixels":0,"materials":{}}
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new_dim = 512
        h, w = img.shape[0:2]
        scale_x = float(new_dim) / float(h)
        scale_y = float(new_dim) / float(w)
        scale = min(scale_x, scale_y)
        new_h = math.ceil(scale * h)
        new_w = math.ceil(scale * w)
        img = Image.fromarray(img).resize((new_w, new_h), Image.Resampling.LANCZOS)
        img = np.array(img)

        image = torch.from_numpy(img.transpose((2, 0, 1))).float()
        image = TTR.Normalize(mean, std)(image)
        if is_cuda:
            image = image.cuda()
        image = image.unsqueeze(0)
        with torch.no_grad():
            prediction = model(image)[0].data.cpu()[0, 0].numpy()
        original_image = img[..., ::-1]
        predicted_colored = apply_color(prediction)
        #predicted_colored = apply_color2(prediction) //test for predicition fetch
        material_mapping = fetch_prediciton(image_path,predicted_colored)
        stacked_img = np.concatenate(
            (np.uint8(original_image), predicted_colored), axis=1
        )
        cv2.imwrite(
            f'{args.output_folder}/{os.path.splitext(os.path.basename(image_path))[0]}.png',
            stacked_img,
        )
    # Test for statistics grab.
    for image_path in images_to_infere:
        print(image_path)
        print(f"Image number of pixels: {images_to_infere[image_path]['num_of_pixels']}")
        for material_index in images_to_infere[image_path]["materials"]:
            print(f"Material name: {images_to_infere[image_path]['materials'][material_index]['name']}")
            print(f"Material color: {images_to_infere[image_path]['materials'][material_index]['rgb_color']}")
            print(f"Material num of pixels: {images_to_infere[image_path]['materials'][material_index]['num_of_pixels']}")
            print("")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--jit_path',
        type=str,
        default='',
        help='path to the pretrained model',
    )
    parser.add_argument(
        '--image_folder',
        type=str,
        default='',
        help='overwrite the data_path for local runs',
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default='',
        help='overwrite the data_path for local runs',
    )
    args = parser.parse_args()
    main(args)
