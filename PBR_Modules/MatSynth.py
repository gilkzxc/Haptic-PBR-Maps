import torchvision.transforms.functional as TF
from datasets import load_dataset
from torch.utils.data import DataLoader
try:
    from .PBR import PBR, tile_maps_keys
except ImportError:
    from PBR import PBR, tile_maps_keys  # Fallback for direct execution
import glob
import json
import argparse

# image processing function        
def process_img(x, scale = 512):
    if x is None:
            return None
    if x.mode == 'I;16':
            x = x.convert('I')
    x = TF.resize(x, (scale, scale))
    if x.mode == 'I':
            x = x.convert('I;16')
    return x

# # item processing function
# def process_batch(examples):
#     examples["basecolor"] = [process_img(x) for x in examples["basecolor"]]
#     return examples



class MatSynth:
    def __init__(self, dataset = None, local_files = None, is_streaming = True):
        # load the dataset in streaming mode
        if not dataset is None:
            self.dataset = dataset
        elif not local_files is None:
            self.dataset = load_dataset('parquet', data_files=local_files, streaming= is_streaming,)
        else:
            self.dataset = load_dataset("gvecchio/MatSynth", streaming = is_streaming,)
            
    
    def filter_by_tags(self, *args):
        ds = self.dataset
        for tag in args:
            ds = ds.filter(lambda x: tag in x["metadata"]["tags"])
        return MatSynth(ds)
    
    #def shuffle(self,buffer_)
    
    def fetch_PBR_train_dict_by_tags(self, tags, number_of_samples = 10):
        temp = list(self.filter_by_tags(*tags).dataset.take(number_of_samples))
        result = {}
        for x in temp:
            result[x['name']] = PBR(x)
        return result
        
if __name__ == '__main__':
    #material_DB_keys = ['Wood', 'Metal', 'Stone', 'Leather', 'Fabric', 'Concrete', 'Ceramic', 'Soil']
    material_DB_keys = ['Wood', 'Metal', 'Fabric']
    MatSynth_test_keys = ['Plastic', 'Ground', 'Metal', 'Concrete', 'Terracotta', 'Misc', 'Wood', 'Ceramic', 'Stone', 'Leather', 'Fabric', 'Marble', 'Plaster']
    key_translator = {"Ground":"Soil", "Soil":"Ground"}
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_files',
        type=str,
        default="/export/MatSynth/test",
        help='path to the test files directory',
    )
    parser.add_argument(
        '--PBR_path',
        type=str,
        default="/export/PBR",
        help='path to the test files directory',
    )
    args = parser.parse_args()
    test_files = glob.glob(f"{args.test_files}/*.parquet")
    MS = MatSynth(local_files={"test":test_files},is_streaming=False)
    ds_test = MS.dataset['test']
    simpler_ds = {key:[] for key in material_DB_keys}
    print("Before filling simpler_ds")
    #for i in range(len(ds_test)):
    for i in range(30,60):
        category = ds_test[i]['metadata']['category']
        if category in simpler_ds:
            simpler_ds[category].append(ds_test[i])
        """elif category in key_translator and key_translator[category] in simpler_ds:
            simpler_ds[key_translator[category]].append(ds_test[i])"""
    """try:
        print(json.dumps(simpler_ds))
    except:
        print("ERROR1")"""
    for category in simpler_ds:
        for i in range(len(simpler_ds[category])):
            p = PBR({key:simpler_ds[category][i][key] for key in tile_maps_keys})
            name = simpler_ds[category][i]['name']
            #simpler_ds[category][i] = p
            p.save(f"{args.PBR_path}/{category}/{name}")

