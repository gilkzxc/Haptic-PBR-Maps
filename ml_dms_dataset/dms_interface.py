# interface with DMS
import inference
import argparse
from ast import If
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
from collections import deque

pretrained_path = "./DMS46_v1.pt"

random.seed(112)
def is_valid_img(img):
    return True

class infered_image:
    def __init__(self,image_path):
        self.image_path = image_path
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
        if inference.is_valid_img(img):
            self.img = img
        else:
            self.img = None
        self.estimated_colors = None
        self.stats = {"num_of_pixels":0,"materials":{}}
        self.material_mapping = []
    
    def was_predicted(self):
        if (not isinstance(self.stats,dict)) or self.stats == {"num_of_pixels":0,"materials":{}}:
            return False
        if (not isinstance(self.material_mapping,list)) or self.material_mapping == []:
            return False
        return True


    def estimation(self,is_cuda,model,mean,std):
        if self.img == None:
            return None
        if self.estimated_colors != None:
            return self.estimated_colors
        image = torch.from_numpy(self.img.transpose((2, 0, 1))).float()
        image = TTR.Normalize(mean, std)(image)
        if is_cuda:
            image = image.cuda()
        image = image.unsqueeze(0)
        with torch.no_grad():
            estimation_label_mask = model(image)[0].data.cpu()[0, 0].numpy()
         
        self.estimated_colors = inference.apply_color(estimation_label_mask)[..., ::-1] #Remember apply_color returns backwards
        return self.estimated_colors
        
    def fetch_estimation(self):
        if self.img == None or self.estimated_colors == None:
            return None
        if self.was_predicted():
            return self.material_mapping
        material_mapping = []
        stats = {"num_of_pixels":0,"materials":{}}
        for row in self.estimated_colors:
            material_mapping.append([])
            for rgb_array in row:
                stats["num_of_pixels"] += 1
                for material_index in range(len(inference.t['srgb_colormap'])):
                    if np.array_equal(rgb_array,inference.t['srgb_colormap'][material_index]):
                        if not material_index in stats["materials"]:
                            stats["materials"][material_index] = {"name":inference.t['names'][material_index],
                                                                                     "rgb_color":inference.t['srgb_colormap'][material_index],
                                                                                    "num_of_pixels":0}
                        stats["materials"][material_index]["num_of_pixels"] += 1
                        material_mapping[-1].append(inference.t['names'][material_index])
                        break
        self.stats = stats
        self.material_mapping = material_mapping
        return material_mapping



class infering_pipeline:
    def __init__(self,model_path,output_folder_path,parameters = inference.parameters):
        self.mean = parameters["mean"]
        self.std = parameters["std"]
        self.output_folder_path = output_folder_path
        self.is_cuda = torch.cuda.is_available()
        self.model = torch.jit.load(model_path)
        if self.is_cuda:
            self.model = self.model.cuda()
        self.pipeline_to_infer = deque([])
        self.pipeline_infered = deque([])
    def insert_into_infer(self, image_path):
        self.pipeline.append(infered_image(image_path))
    def run_pipeline(self):
        #something
        os.makedirs(self.output_folder_path, exist_ok=True)
        while len(self.pipeline_to_infer) > 0:
            head = self.pipeline_to_infer.popleft()
            estimation_color = head.estimation(self.is_cuda,self.model,self.mean,self.std)
            material_mapping = head.fetch_estimation()
            self.pipeline_infered.append(head)
            stacked_img = np.concatenate((np.uint8(head.img), estimation_color), axis=1)
            cv2.imwrite(f'{self.output_folder_path}/{os.path.splitext(os.path.basename(head.image_path))[0]}.png',stacked_img,)
