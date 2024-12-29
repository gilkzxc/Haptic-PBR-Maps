from diffusers import DiffusionPipeline, LCMScheduler, UNet2DConditionModel

from PBR import PBR
from diffusers.utils import load_image
import torch
from os import path
from PIL import Image



class PBR_Diffuser:
    def __init__(self, is_consistent = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unet = None
        if is_consistent:
            self.unet = UNet2DConditionModel.from_pretrained("gvecchio/StableMaterials",subfolder="unet_lcm",torch_dtype=torch.float16,)
            self.pipe = DiffusionPipeline.from_pretrained("gvecchio/StableMaterials", trust_remote_code=True, unet=self.unet, torch_dtype=torch.float16)
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
            self.num_inference_steps = 4
        else:
            self.pipe = DiffusionPipeline.from_pretrained("gvecchio/StableMaterials", trust_remote_code=True, torch_dtype=torch.float16)
            self.num_inference_steps = 50

        self.pipe.to(self.device)
    def generator(self, prompt):
        try:
            # Tring prompt as local image file path or URL to image.
            prompt = load_image(prompt)
        except ValueError:
            # Prompt is open text.
            prompt = prompt
        material = self.pipe(prompt=prompt,guidance_scale=10.0,tileable=True,num_images_per_prompt=1,num_inference_steps=self.num_inference_steps,).images[0]
        try:
            result = PBR({'basecolor':material.basecolor,'normal':material.normal,'height':material.height,
                         'roughness':material.roughness, 'metallic':material.metallic})
        except:
            result = None
        return result

    
