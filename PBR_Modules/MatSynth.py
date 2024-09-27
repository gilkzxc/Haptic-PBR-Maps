import torchvision.transforms.functional as TF
from datasets import load_dataset
from torch.utils.data import DataLoader
from PBR import PBR


# image processing function
def process_img(x, scale = 512):
    x = TF.resize(x, (scale, scale))
    x = TF.to_tensor(x)
    return x

# item processing function
def process_batch(examples):
    examples["basecolor"] = [process_img(x) for x in examples["basecolor"]]
    return examples

        
        


class MatSynth:
    def __init__(self, dataset = None, is_streaming = True):
        # load the dataset in streaming mode
        if dataset is None:
            self.dataset = load_dataset("gvecchio/MatSynth", streaming = is_streaming,)
        else:
            self.dataset = dataset
     
    def preprocess():
        return

    #def 
    
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
        
