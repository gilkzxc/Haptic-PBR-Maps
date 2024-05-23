#main
from collections import deque
from genericpath import isfile
import ml_dms_dataset
import PBR_Modules
import argparse
from os import path
from glob import glob
import questionary
Types = ["text","image","PBR"]
States = {"PBR":"PBR transform", "MS":"Material Segmentation", "MP":"Material Properties", "Haptic":"Haptic Transform"}

class Task:
    def __init__(self,initial_type,file_path,data = None, init_state = None):
        self.type = initial_type
        self.original_file_path = file_path
        self.data = data
        self.nextState = init_state
        
    

Tasks = []


        

def PBR_transform():
    #
    return ""

def matching_PBR_data_by_raw_input_fetcher():
    #runs ONE-PEACE along our included huge PBR datasets. Retrieve the best matching PBR.
    return ""

def material_segmentation(pretrained_path,input_folder,output_folder):
    if Tasks is None:
        return None
    #Runs Material Segmentation model
    dms_pipeline = ml_dms_dataset.infering_pipeline(pretrained_path,output_folder)
    for task in Tasks:
        if task.nextState == States["MS"]:
            dms_pipeline.insert_into_infer(input_folder+task.file_path)
    dms_pipeline.run_pipeline()
    results_pipeline = dms_pipeline.pipeline_infered
    while len(results_pipeline) > 0:
        head = results_pipeline.popleft()
        

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
            result = {}
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
    
def main():
    print("Welcome to Haptic PBR Generator")
    while run:
        fetch_and_order_input()
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()