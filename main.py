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
from Materials_DB.material_properties import load_properties_json

Types = ["text","image","PBR","rendered_PBR"]
            
            


        

def yes_no_question(question):
    return questionary.select(question,choices=["Yes","No"]).ask()


    

def runCycle(tasks_pipe, dms_pipeline, sm_diffuser, mf_diffuser):
    new_pipe = deque([])
    while len(tasks_pipe) > 0:
        head = tasks_pipe.popleft()
        if tasks.States[head.nextState] == "PBR transform":
            # Run in PBR diffusers.
            run_pbr_successfull = head.to_PBR(sm_diffuser, mf_diffuser)
            if not run_pbr_successfull:
                head.deleteOutput()
                continue
        elif tasks.States[head.nextState] == "Material Segmentation":
            # Run in dms segmentation.
            run_material_segmentation_successfull = head.to_material_segmentation(dms_pipeline)
            if not run_material_segmentation_successfull:
                head.deleteOutput()
                continue
        elif tasks.States[head.nextState] == "Material Properties":
            # Run in dms segmentation.
            run_material_properties_successfull = head.to_material_properties()
            if not run_material_properties_successfull:
                head.deleteOutput()
                continue
        elif tasks.States[head.nextState] == "Haptic Transform":
            # Export into Haptic PBR file format .hpbr
            head.to_HapticTransform()

        head.nextState += 1
        if head.nextState < len(tasks.States): # Still not finished.
          new_pipe.append(head)
    return new_pipe


def getPrompt(output_folder):
    prompt_type = questionary.select("Type of prompt:",choices=tasks.prompt_types).ask()
    if prompt_type == tasks.prompt_types[0] or prompt_type == tasks.prompt_types[1]:
        file_folder_path = questionary.path("Enter path: ").ask()
        return tasks.Task(prompt_input=file_folder_path,output_parent_dir=output_folder)
    elif prompt_type == tasks.prompt_types[2]:
        url_path = input("Enter url path: ")
        return tasks.Task(prompt_input=url_path,output_parent_dir=output_folder)
    
    # Free text
    free_text = input("Enter text: ")
    return tasks.Task(prompt_input=free_text,output_parent_dir=output_folder)
        
def main(args):
    print("Welcome to Haptic PBR Generator")
    load_properties_json(args.properties_json)
    run = True
    dms_pipeline = ml_dms_dataset.infering_pipeline(args.pretrained_dms_path)
    sm_diffuser = StableMaterials.PBR_Diffuser()
    #mf_diffuser = MatForger.PBR_Diffuser()
    mf_diffuser = None
    os.makedirs(args.output_folder, exist_ok=True)
    Tasks = deque([])
    while run:
        if yes_no_question("Skip prompt?") == "No":
            Tasks.append(getPrompt(output_folder=args.output_folder))
        Tasks = runCycle(Tasks, dms_pipeline, sm_diffuser, mf_diffuser)
        run = len(Tasks) > 0
    print("Exiting Haptic PBR Generator")
    

if __name__ == '__main__':
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(prog='Haptic-PBR-Maps Generator', description='Utilize material segmentation with PBR diffusers to create Haptic PBR Maps.')
    parser.add_argument(
        '--pretrained_dms_path',
        type=str,
        default=f'{current_dir}/ml_dms_dataset/DMS46_v1.pt',
        help='path to the pretrained model of DMS',
    )
    parser.add_argument(
        '--properties_json',
        type=str,
        default=f'{current_dir}/Material_DB/material_DB.json',
        help='path to the properties DB',
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default=f'{current_dir}/output/',
        help='path to output folder',
    )
    """parser.add_argument(
        '--consistent',
        type=str,
        default='./output/',
        help='Run Consistent StableMaterial',
    )"""
    main(parser.parse_args())
