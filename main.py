#main
import ml_dms_dataset
import argparse
Types = ["text","sound","image","PBR"]

class Task:
    def __init__(self,initial_type,file_path,data):
        self.type = initial_type
        self.original_file_path = file_path
        self.data = data
        self.nextState = None
    
    def 
        
    
def fetch_and_order_input():
    #Organise the input from user, wheter from GUI or CLI mode.
    return {}

def PBR_transform_generator():
    #runs EasyPBR, in order to create a PBR correspondor files for each image from input.
    return ""

def matching_PBR_data_by_raw_input_fetcher():
    #runs ONE-PEACE along our included huge PBR datasets. Retrieve the best matching PBR.
    return ""

def material_segmentation():
    #Runs Material Segmentation model
    
def main():
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()