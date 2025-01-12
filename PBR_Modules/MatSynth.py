import torchvision.transforms.functional as TF
from datasets import load_dataset
from torch.utils.data import DataLoader
from .PBR import PBR


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
        
