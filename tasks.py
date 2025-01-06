from diffusers.utils import load_image
from os.path import isdir, isfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from glob import glob
#from text_to_image import genImage
from gradio_client import Client

def is_valid_url(s):
    try:
        result = urlparse(s)
        if not all([result.scheme, result.netloc]):
            return False, "Invalid URL structure"
    except ValueError:
        return False, "Invalid URL structure"
    try:
        req = Request(s, method="HEAD")
        with urlopen(req, timeout=5) as response:
            if response.status == 200:
                return True, "URL is working"
            else:
                return False, f"URL returned status code {response.status}"
    except HTTPError as e:
        return False, f"HTTP error occurred: {e.code}"
    except URLError as e:
        return False, f"URL error occurred: {e.reason}"
    except Exception as e:
        return False, f"An unexpected error occurred: {e}"

def genImage(prompt_input, negative_prompt_input = "", width = 512, height = 512, guidance_scale = 4.5,num_inference_steps=40):
    client = Client("stabilityai/stable-diffusion-3.5-large")
    result = client.predict(
		    prompt=prompt_input,
		    negative_prompt=negative_prompt_input,
		    seed=0,
		    randomize_seed=True,
		    width=width,
		    height=height,
		    guidance_scale=guidance_scale,
		    num_inference_steps=num_inference_steps,
		    api_name="/infer"
    )
    print(result)
    file_path = result[0]
    return file_path


prompt_types = ["Folder Path", "File Path", "Url Path", "Free Text"]
class prompt:
    def __init__(self, prompt_input, prompt_type = None):
        if not isinstance(prompt_input, str):
            raise TypeError(f"Prompt must be a string. Your input: {prompt_input} and it's type: {type(prompt_input)}")
        self.value = prompt_input
        self.prompt_type = prompt_type
        if prompt_type is None:
            if isdir(prompt_input):
                self.type = prompt_types[0]
            elif isfile(prompt_input):
                self.type = prompt_types[1]
            else:
                is_valid, message = is_valid_url(prompt_input)
                if is_valid:
                    self.prompt_type = prompt_types[2]
                elif message == "Invalid URL structure":
                    self.prompt_type = prompt_types[3]
                else:
                    raise ValueError(f"Prompt was recognised as URL address, but this error occured: {message}")




    
States = {"PBR":"PBR transform", "MS":"Material Segmentation", "MP":"Material Properties", "Haptic":"Haptic Transform"}
        
class Task:
    def __init__(self,prompt_input, init_state = None):
        #self.type = initial_type
        self.prompt_text = None
        self.prompt_image = None
        self.children = []
        self.material_segmentation = None
        self.PBR = None
        self.nextState = init_state
        try:
            self.prompt_text = prompt(prompt_input)
        except ValueError as e:
            print(f"A ValueError occurred: {e}")
        except TypeError as e:
            print(f"A TypeError occurred: {e}")
        if self.prompt_text:
            if self.prompt_text.type == prompt_types[0]:
                # Prompt is folder of files
                # Firstly we will allow only one redirection
                list_of_file_paths = glob(f'{self.prompt_text.value}/*')
                self.children = [Task(file_path) for file_path in list_of_file_paths if isfile(file_path)]
            elif self.prompt_text.type == prompt_types[1] or self.prompt_text.type == prompt_types[2]:
                # Prompt is file path or url.
                try:
                    image = load_image(self.prompt_text.value)
                    self.prompt_image = image_file_path
                except ValueError as e:
                    print(f"A ValueError occurred: {e}")
                    self.prompt_image = None
            else:
                # Prompt is free text. So we will generate an image.
                image = load_image(genImage(self.prompt_text.value))
                self.prompt_image = image_file_path
        

        
        
