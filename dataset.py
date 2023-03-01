import os
import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
class dataset(Dataset):
    def __init__(self,dataset_dir = 'dataset',transform = False):
        self.dataset_dir = dataset_dir
        self.transforms = transform if transform else transforms.Compose([
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.double),
                    transforms.Resize((224,224)),
                    ])

    def __name_files__(self):
        return [file_name for file_name in os.listdir(f'{self.dataset_dir}/Images')]
    
    def __len__(self):
        return len(self.__name_files__())

    def __getitem__(self, idx):
       file_name = self.__name_files__()[idx]
       img = Image.open(f'{self.dataset_dir}/Images/{file_name}')
       return file_name,self.transforms(img)
