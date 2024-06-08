from numpy import imag
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from PBR import PBR
from collections import deque
from os import path

class PBR_diffusion_pipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = StableDiffusionPipeline.from_pretrained("gvecchio/MatForger",trust_remote_code=True,)
        self.pipe.enable_vae_tiling()
        self.pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2)
        self.pipe.to(self.device)
        self.pipeline_to_diffuse = deque([])
        
    def PBR_textures_map_generator(self,prompt):
        if path.isfile(prompt):
            prompt = Image.open(prompt)
        image = self.pipe(prompt,guidance_scale=6.0,height=512,width=512,
                tileable=True, # Allows to generate tileable materials
                patched=False, # Reduce memory requirements for high-hes generation but affects quality 
                num_inference_steps=25,).images[0]
        try:
            result = PBR({'basecolor':image.basecolor,'normal':image.normal,'height':image.height,
                         'roughness':image.roughness, 'metallic':image.metallic})
        except:
            result = None
        return result
    