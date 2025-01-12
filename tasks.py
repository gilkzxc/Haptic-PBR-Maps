from diffusers.utils import load_image
from os.path import isdir, isfile, basename, dirname
from os import makedirs
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from glob import glob
#from text_to_image import genImage
from gradio_client import Client
from shutil import rmtree


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
        self.type = prompt_type
        if prompt_type is None:
            if isdir(prompt_input):
                self.type = prompt_types[0]
            elif isfile(prompt_input):
                self.type = prompt_types[1]
            else:
                is_valid, message = is_valid_url(prompt_input)
                if is_valid:
                    self.type = prompt_types[2]
                elif message == "Invalid URL structure":
                    self.type = prompt_types[3]
                else:
                    raise ValueError(f"Prompt was recognised as URL address, but this error occured: {message}")



States = ["PBR transform","Material Segmentation","Material Properties","Haptic Transform"]

        
class Task:
    def __init__(self, prompt_input, output_parent_dir = "./", init_state = 0):
        #self.type = initial_type
        self.prompt_text = None
        self.prompt_image = None
        self.output_parent_dir = output_parent_dir
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
                self.output_dir = f"{self.output_parent_dir}/{basename(self.prompt_text.value)}"
                makedirs(self.output_dir, exist_ok=True)
                # Firstly we will allow only one redirection
                list_of_file_paths = glob(f'{self.prompt_text.value}/*')
                self.children = [Task(file_path,output_parent_dir=self.output_dir) for file_path in list_of_file_paths if isfile(file_path)]
            else:
                try:
                    if self.prompt_text.type == prompt_types[-1]:
                        # Prompt is free text.
                        base = basename(self.prompt_text.value)
                        base = base+"_" if base[-1] == "." else base # Free text can end in point, which in some os like Windows, cause bugs in path.
                        self.output_dir = f"{self.output_parent_dir}/{base}"
                        makedirs(self.output_dir, exist_ok=True)
                        # So we will generate an image.
                        image = load_image(genImage(self.prompt_text.value))
                        image_file_path = f"{self.output_dir}/prompt_text_to_prompt_image.png"
                        image.save(image_file_path)
                    else:
                        # Prompt is file path or url.
                        self.output_dir = f"{self.output_parent_dir}/{basename(self.prompt_text.value)}"
                        makedirs(self.output_dir, exist_ok=True)
                        image = load_image(self.prompt_text.value)
                        image_file_path = self.prompt_text.value
                    self.prompt_image = {"path":image_file_path, "image":image}
                except ValueError as e:
                    print(f"A ValueError occurred: {e}")
                    self.prompt_image = None
    def isTaskDir(self):
        return (self.prompt_text is not None) and (self.prompt_text.type == prompt_types[0])
    def isTaskImage(self):
        return (self.prompt_text is not None) and (self.prompt_text.type == prompt_types[1] or self.prompt_text.type == prompt_types[2]) and (self.prompt_image is not None)
    def isTaskFreeText(self):
        return (self.prompt_text is not None) and (self.prompt_text.type == prompt_types[-1]) and (self.prompt_image is not None)
    def deleteOutput(self):
        try:
            rmtree(self.output_dir)
            print(f'Folder {self.output_dir} and its content removed') # Folder and its content removed
        except:
            print(f'Folder {self.output_dir} not deleted')
    def to_material_segmentation(self,dms_pipeline):
        print(f"Prompt: {self.prompt_text.value} , begins Material Segmentation.")
        if self.isTaskDir():
            if self.children:
                print(f"Begin children uploading to pipeline.")
                for child in self.children:
                    print(f"Child path: {child.prompt_text.value}")
                    if child.isTaskImage() or child.isTaskFreeText():
                        dms_pipeline.insert_into_infer(child.prompt_image["path"],f"{child.output_dir}/dms")
                    else:
                        print("Error in child. Deleting parent and it's children output.")
                        return False
                print("Begin workload...")
                if dms_pipeline.run_pipeline():
                    print("Passing workload to children")
                    for child_index in range(len(self.children)):
                        self.children[child_index] = dms_pipeline.pipeline_infered.popleft()
                    print("Done...")
                    return True
                return False
            else:
                print(f"{self.prompt_text.value} is an empty directory.")
                return False
        elif self.isTaskImage() or self.isTaskFreeText():
            run_singleton_result = dms_pipeline.run_singleton(self.prompt_image["path"],f"{self.output_dir}/dms")
            if run_singleton_result is None:
                print(f"ERROR: {self.prompt_text.value} singleton run failed.")
                return False
            self.material_segmentation = run_singleton_result
            print("Done...")
            return True
        return False
    def to_PBR(self,sm_diffuser, mf_diffuser):
        print(f"Prompt: {self.prompt_text.value} , begins PBR tile maps generation.")
        if self.isTaskDir():
            if self.children:
                print(f"Begin children uploading to pipeline.")
                for child in self.children:
                    print(f"Child path: {child.prompt_text.value}") 
                    if not child.to_PBR(sm_diffuser, mf_diffuser):
                        print("Error in child. Deleting parent and it's children output.")
                        return False
                print("Done children...")
            else:
                print(f"{self.prompt_text.value} is an empty directory.")
                return False
        elif self.isTaskImage():
            sm_gen = {}
            if not self.prompt_image is None:
                sm_gen["image"] = sm_diffuser.generator(self.prompt_image["path"])
                if not sm_gen["image"] is None:
                    sm_gen["image"].save(f"{self.output_dir}/sm/image")
                else:
                    print(f"Prompt: {self.prompt_text.value} , Error StableMaterials failed in image prompt.")
            else:
                print(f"Prompt: {self.prompt_text.value} , Error StableMaterials failed in image prompt.")
            if sm_gen != {} and sm_gen["image"] is not None:
                if self.PBR is None:
                    self.PBR = {}
                self.PBR["SM"] = sm_gen
                print("Done...")
                return True
            return False
        elif self.isTaskFreeText():
            sm_gen = {}
            if not self.prompt_image is None:
                sm_gen["image"] = sm_diffuser.generator(self.prompt_image["path"])
                if not sm_gen["image"] is None:
                    sm_gen["image"].save(f"{self.output_dir}/sm/image")
                else:
                    print(f"Prompt: {self.prompt_text.value} , Error StableMaterials failed in image prompt.")
            else:
                print(f"Prompt: {self.prompt_text.value} , Error StableMaterials failed in image prompt.")
            sm_gen["text"] = sm_diffuser.generator(self.prompt_text.value)
            if not sm_gen["text"] is None:
                sm_gen["text"].save(f"{self.output_dir}/sm/text")
            if sm_gen != {} and ("image" in sm_gen and sm_gen["image"] is not None) and ("text" in sm_gen and sm_gen["text"] is not None):
                if self.PBR is None:
                    self.PBR = {}
                self.PBR["SM"] = sm_gen
                print("Done...")
                return True
            return False

        
        
