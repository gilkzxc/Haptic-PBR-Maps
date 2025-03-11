#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# This code accompanies the research paper: Upchurch, Paul, and Ransen
# Niu. "A Dense Material Segmentation Dataset for Indoor and Outdoor
# Scene Parsing." ECCV 2022.
#
# This example shows how to predict materials.
#

import json
import numpy as np
import matplotlib.pyplot as plt
import os

key_translator = {"Fabric\ncloth":"Fabric", "Soil/mud":"Soil", "Stone\nnatural":"Stone", "Stone\npolished":"Stone", "Wood\ntree":"Wood"}

def load_properties_json(db_path):
    global material_properties_dict
    with open(db_path, 'r') as file:
        material_properties_dict = json.load(file)

class Property:
    def __init__(self, name, category, label):
        self.name = name
        self.category = category
        self.label = label

    def __repr__(self):
        return f"Property(name='{self.name}', category='{self.category}', label='{self.label}')"


# Predefined list of supported properties
supported_properties = [
    Property("Young\'s Modulus", "Tactile Compliance", "YoungsModulus"),
    Property("Poisson\'s Ratio", "Tactile Compliance", "PoissonRatio"),
    Property("Coefficient of Static Friction", "Tactile Stiction", "Coefficient of Static Friction"),
    Property("Coefficient of Kinetic Friction", "Sliding Resistance", "Coefficient of Kinetic Friction"),
    Property("Roughness Parameters", "Macro Roughness", "Roughness Parameters"),
    Property("Thermal Effusivity", "Thermal Cooling", "Thermal Effusivity"),
    Property("Thermal Diffusivity", "Thermal Persistence", "Thermal Diffusivity"),
    Property("Thermal Conductivity", "Thermal Conductivity", "Thermal Conductivity")
]

material_properties_dict = None

def create_maps(predicted_map):
    dict={}
    for property in supported_properties:
        dict[property.label] = create_property_map(predicted_map,property.category,property.name)
    return dict

def create_property_map(pixel_array, type, name):
    if isinstance(pixel_array, list):
        pixel_array = np.array(pixel_array)
    new_array = np.zeros_like(pixel_array, dtype=float)
    rows, cols = pixel_array.shape
    for i in range(rows):
        for j in range(cols):
            if pixel_array[i, j] in material_properties_dict:
                characteristics = material_properties_dict[pixel_array[i, j]]
                new_array[i, j] = characteristics[type][name]["value"]
            elif pixel_array[i, j] in key_translator:
                characteristics = material_properties_dict[key_translator[pixel_array[i, j]]]
                new_array[i, j] = characteristics[type][name]["value"]
            else:
                new_array[i, j] = np.nan
    return new_array


def print_map(map):
    # Plot grayscale maps
    plt.figure(figsize=(12, 6))
    i=1
    cmap = plt.cm.gray_r
    cmap.set_bad(color='lightcoral')
    for key, value in map.items():
        plt.subplot(2, 4, i)
        i=i+1
        plt.imshow(value, cmap=cmap, aspect='auto')
        plt.title(key)
        plt.colorbar(label=key)

    plt.tight_layout()
    plt.show()


def save_map_as_png(prop_maps, file_name,output_dir='output'):
    import os
    cmap = plt.cm.gray_r
    cmap.set_bad(color='lightcoral')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(prop_maps)
    for i, (key, value) in enumerate(prop_maps.items(), start=1):
        plt.figure(figsize=(6, 6))
        plt.imshow(value, cmap=cmap, aspect='auto')
        plt.title(key)
        plt.colorbar(label=key)

        # Save each figure as a PNG file
        filename = os.path.join(output_dir, f"{file_name}_{key}.png")
        plt.savefig(filename, dpi=300)
        plt.close()  # Close the figure to free memory
        print(f"Saved: {filename}")

    print("All maps saved successfully.")

def run_material_properties(predicted_map,output_dir,file_name):
    output_maps=create_maps(predicted_map)
    file_base_name=os.path.splitext(os.path.basename(file_name))[0]
    save_map_as_png(output_maps,file_base_name,output_dir)
    #print_map(output)
    return output_maps

