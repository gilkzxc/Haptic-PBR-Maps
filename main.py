#main
from collections import deque

from PBR_Modules import MatForger, MatSynth, StableMaterials
import ml_dms_dataset
import PBR_Modules
import argparse


import questionary
import cv2
import os
#from tasks import Task, prompt_types
import tasks

Types = ["text","image","PBR","rendered_PBR"]


class Task:
    def __init__(self,initial_type,file_path,data = None, init_state = None):
        self.type = initial_type
        self.original_file_path = file_path
        self.data = data
        self.nextState = init_state
        
    def to_PBR(self, PBR_diffuser, dataset_mode = False):
        if self.nextState != States["PBR"]:
            return None
        generated_PBR_obj = None
        if self.type == "image":
            generated_PBR_obj = PBR_diffuser.PBR_textures_map_generator(self.original_file_path)
        elif self.type == "text":
            generated_PBR_obj = PBR_diffuser.PBR_textures_map_generator(self.data)
        if (not generated_PBR_obj is None) and (not generated_PBR_obj.to_render() is None):
            self.data = generated_PBR_obj
            self.type = "rendered_PBR"
            self.nextState = States["MS"]
            return self.data
        return None
    def material_segmentation(self,dms_pipeline,input_folder_path):
        if self.nextState != States["MS"] or self.type != "rendered_PBR":
            return None
        run_singleton_result = dms_pipeline.run_singleton(infered_image(image_path,output_folder_path))
        if run_singleton_result is None:
            print("ERROR")
        
        
    def action(self, PBR_diffuser, dms_pipeline, input_folder_path = ""):
        if self.nextState == States["PBR"]:
            if self.to_PBR(PBR_diffuser) is None:
                print("ERROR")
                return False
            PBR_output_folder = f"/output/{input_folder_path}/PBR/"
            os.makedirs(PBR_output_folder, exist_ok=True)
            cv2.imwrite(f'{PBR_output_folder/self.original_file_path}',cv2.cvtColor(self.data.render, cv2.COLOR_RGB2BGR),)
            
            return True
        elif self.nextState == States["MS"]:
            ms_result = self.material_segmentation(dms_pipeline,input_folder_path)
            if ms_result is None:
                print("ERROR")
                return False
            ms_
            
            
Tasks = deque([])



def material_segmentation(task,dms_pipeline):
    #Runs Material Segmentation model
    if isinstance(task,dict):
        for file_path in task:
            if task[file_path].nextState == States["MS"]:
                if task[file_path].to_PBR(PBR_diffuser) is None:
                    print("ERROR")
    elif isinstance(task, Task):
        if task.nextState == States["MS"]:
            if task.to_PBR(PBR_diffuser) is None:
                print("ERROR")
        

def yes_no_question(question):
    return questionary.select(question,choices=["Yes","No"]).ask()


def fetch_and_order_input():
    #Organise the input from user, from CLI mode.
    result = None
    init_state = States["PBR"]
    if questionary.select("Type of prompt:",choices=['Free text','File/Folder Path']).ask() == "File/Folder Path":
        file_folder_path = questionary.path("Enter file or folder path: ").ask()
        type_of_input = "image"
        if path.isdir(file_folder_path): 
            list_of_file_paths = glob(f'{file_folder_path}/*')
            result = {"input_folder_path":file_folder_path}
            if yes_no_question("Is it a folder of rendered PBR items?") == "Yes":
                type_of_input = "rendered_PBR"
                init_state = States["MS"]
            for file_path in list_of_file_paths:
                result[file_path] = Task(initial_type=type_of_input, file_path=file_path, init_state=init_state)
        elif path.isfile(file_folder_path):
            if yes_no_question("Is it a rendered PBR?") == "Yes":
                type_of_input = "rendered_PBR"
                init_state = States["MS"]
            result = Task(initial_type=type_of_input, file_path=file_folder_path, init_state=init_state)
        else:
            print("This is not a valid path of an existen file or folder.")
            return
    else:
        free_text = input("Enter free text prompt: ")
        result = Task(initial_type="text", data=free_text, init_state=init_state)
    Tasks.append(result) 
    
"""def main():
    print("Welcome to Haptic PBR Generator")
    run = True
    while run:
        fetch_and_order_input()
        PBR_diffuser = PBR_Modules.PBR_diffusion_pipeline()
        PBR_dataset = MatSynth()
        dms_pipeline = ml_dms_dataset.infering_pipeline(pretrained_path)
        for task in Tasks:  #while Tasks != []
            if isinstance(task,dict):
                for file_path in task:
                    if task[file_path].action(PBR_diffuser,dms_pipeline,task["input_folder_path"]):
                        
                
            elif isinstance(task, Task):
                if task.action(PBR_diffuser,dms_pipeline):"""

def runCycle(tasks_pipe, dms_pipeline, sm_diffuser, mf_diffuser):
    new_pipe = deque([])
    while len(tasks_pipe) > 0:
        head = tasks_pipe.popleft()
        if create_pbr_tile_maps:
            # Run in PBR diffusers.
        elif run_material_segmentation:
            # Run in dms segmentation.
            print(f"Prompt: {head.prompt_text.value} , begins Material Segmentation.")
            if head.isTaskDir():
                if head.children:
                    print(f"Begin children uploading to pipeline.")
                    error_for = False
                    for child in head.children:
                        print(f"Child path: {child.prompt_text.value}")
                        if child.isTaskImage() or child.isTaskFreeText():
                            dms_pipeline.insert_into_infer(child.prompt_image["path"],f"{child.output_dir}/dms")
                        else:
                            print("Error in child. Deleting parent and it's children output.")
                            error_for = True
                            break
                    if error_for:
                        head.deleteOutput()
                        continue
                    print("Begin workload...")
                    dms_pipeline.run_pipeline()
                    print("Done...")
                else:
                    print(f"{head.prompt_text.value} is an empty directory.")
            elif head.isTaskImage() or head.isTaskFreeText():
                run_singleton_result = dms_pipeline.run_singleton(ml_dms_dataset.infered_image(head.prompt_image["path"],f"{head.output_dir}/dms"))
                if run_singleton_result is None:
                    print("ERROR")

        elif run_haptic_props:
            # Run in 
        if not head.is_done:
            new_pipe.append(head)
    return new_pipe


def getPrompt():
    prompt_type = questionary.select("Type of prompt:",choices=tasks.prompt_types).ask()
    if prompt_type == tasks.prompt_types[0] or prompt_type == tasks.prompt_types[1]:
        file_folder_path = questionary.path("Enter path: ").ask()
        return tasks.Task(file_folder_path)
    elif prompt_type == tasks.prompt_types[2]:
        url_path = input("Enter url path: ")
        return tasks.Task(url_path)
    
    # Free text
    free_text = input("Enter text: ")
    return tasks.Task(free_text)
        
def main(args):
    print("Welcome to Haptic PBR Generator")
    run = True
    dms_pipeline = ml_dms_dataset.infering_pipeline(args.pretrained_dms_path)
    sm_diffuser = StableMaterials.PBR_Diffuser()
    mf_diffuser = MatForger.PBR_Diffuser()
    os.makedirs(args.output_folder, exist_ok=True)
    while run:
        if yes_no_question("Skip prompt?") == "No":
            Tasks.append(getPrompt())
        Tasks = runCycle(Tasks, dms_pipeline, sm_diffuser, mf_diffuser)
        run = len(Tasks) > 0
    print("Exiting Haptic PBR Generator")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Haptic-PBR-Maps Generator', description='Utilize material segmentation with PBR diffusers to create Haptic PBR Maps.')
    parser.add_argument(
        '--pretrained_dms_path',
        type=str,
        default='./ml_dms_datatset/DMS46_v1.pt',
        help='path to the pretrained model of DMS',
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default='./output/',
        help='path to output folder',
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default='./output/',
        help='path to output folder',
    )
    """parser.add_argument(
        '--consistent',
        type=str,
        default='./output/',
        help='Run Consistent StableMaterial',
    )"""
    main(parser.parse_args())
