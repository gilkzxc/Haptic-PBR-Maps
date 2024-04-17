# interface with DMS
import inference
import argparse
from ast import If, Try
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
import matplotlib.pyplot as plt

pretrained_path = "./DMS46_v1.pt"

random.seed(112)
def is_valid_img(img):
    return True

def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'
    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]
    a = np.asarray( a, dtype='float32' ) / 255.0
    R, G, B = background
    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B
    return np.asarray( rgb, dtype='uint8' )

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
        if is_valid_img(img):
            self.img = img
        else:
            self.img = None
        self.estimated_colors = None
        self.stats = {"num_of_pixels":0,"materials":{}}
        self.material_mapping = []
        self.histogram_img_plot = None
    
    def was_predicted(self):
        if (not isinstance(self.stats,dict)) or self.stats == {"num_of_pixels":0,"materials":{}}:
            return False
        if (not isinstance(self.material_mapping,list)) or self.material_mapping == []:
            return False
        return True


    def estimation(self,is_cuda,model,mean,std):
        if not isinstance(self.img,np.ndarray):
            return None
        if isinstance(self.estimated_colors,np.ndarray):
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
        if not isinstance(self.img,np.ndarray) or not isinstance(self.estimated_colors,np.ndarray):
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
    
    def histogram(self):
        if not self.was_predicted():
            return None
        # Extracting RGB values and counts of the different materials.
        names = [self.stats['materials'][material_index]['name'] for material_index in self.stats["materials"]]
        colors = [self.stats['materials'][material_index]['rgb_color'] for material_index in self.stats["materials"]]
        counts = [self.stats['materials'][material_index]['num_of_pixels'] for material_index in self.stats["materials"]]
        
        # Plotting histogram
        fig, ax = plt.subplots()
        fig.set_size_inches((d/fig.dpi for d in self.img.shape[:2][::-1]))
        ax.bar(names, counts, color=[(r / 255, g / 255, b / 255) for (r, g, b) in colors])
        ax.set_ylabel('pixel count')
        ax.set_xlabel('Materials')
        ax.set_title('Pixel histogram for each material')
        #fig.set_size_inches((d/fig.dpi for d in self.img.shape[:2][::-1]))
        plt.show()
        fig.canvas.draw()
        rgb = rgba2rgb(np.array(fig.canvas.buffer_rgba()))
        cv2.imwrite(f'~/output3/{os.path.splitext(os.path.basename(self.image_path))[0]}.png'
                    ,rgb[..., ::-1],)
        self.histogram_img_plot = rgb
        #self.histogram_img_plot = rgba2rgb(np.array(fig.canvas.buffer_rgba()))
        return self.histogram_img_plot
        
    
    def write_concate_results(self,output_folder_path):
        #OpenCV Works in a BGR format. Thus we need to flip.
        if self.was_predicted():
            #original_img = np.uint8(self.img[..., ::-1])
            original_img = self.img
            if isinstance(original_img,np.ndarray):
                #estimated_colors = self.estimated_colors[..., ::-1]
                estimated_colors = self.estimated_colors
                if isinstance(estimated_colors,np.ndarray):
                    #histogram = self.histogram_img_plot[..., ::-1]
                    histogram = self.histogram_img_plot
                    if isinstance(histogram,np.ndarray):
                        stacked_img = np.concatenate((original_img, estimated_colors,histogram), axis=1)
                        #cv2.imwrite(f'{output_folder_path}/{os.path.splitext(os.path.basename(self.image_path))[0]}.png',stacked_img,)
                        cv2.imwrite(f'{output_folder_path}/{os.path.splitext(os.path.basename(self.image_path))[0]}.png',cv2.cvtColor(stacked_img, cv2.COLOR_RGB2BGR),)
            
            

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
        self.pipeline_to_infer.append(infered_image(image_path))
    def run_pipeline(self):
        #something
        os.makedirs(self.output_folder_path, exist_ok=True)
        while len(self.pipeline_to_infer) > 0:
            head = self.pipeline_to_infer.popleft()
            print(f"Image {head.image_path} : Running Material Segmentation model...")
            estimation_color = head.estimation(self.is_cuda,self.model,self.mean,self.std)
            material_mapping = head.fetch_estimation()
            print(f"Image {head.image_path} stats:\n{head.stats}")
            np.histogram = head.histogram()
            head.write_concate_results(self.output_folder_path)
            print(f"Image {head.image_path} : Saved result in desired output file.")
            self.pipeline_infered.append(head)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pretrained_path',
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
    images_list = glob.glob(f'{args.image_folder}/*')
    print("Initiating pipelines.")
    test_pipe = infering_pipeline(args.pretrained_path,args.output_folder)
    print("Inserting images into pipelines.")
    for image_path in images_list:
        print(image_path)
        test_pipe.insert_into_infer(image_path)
    print("Begin workload...")
    test_pipe.run_pipeline()
    print("Done...")
    
