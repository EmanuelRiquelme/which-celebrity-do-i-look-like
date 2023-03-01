import torch
import torch.nn as nn
import torch.nn.functional as F
from vgg_face import VGG_16
from torchvision import transforms
from PIL import Image
from dataset import dataset
from torch.utils.data import DataLoader
import gc

def inference(data,model,img,cos = nn.CosineSimilarity(dim=1, eps=1e-6)):
    it = iter(data)
    names,sim = [],[]
    with torch.no_grad():
        for _ in range(len(data)):
            celeb_name,celeb_tensor = next(it)
            celeb_tensor = celeb_tensor.to('cuda')
            celeb_tensor = model(celeb_tensor)
            cos_sim =  cos(img,celeb_tensor)
            cos_sim_idx = torch.argmax(cos_sim).item()
            names.append(celeb_name[cos_sim_idx])
            sim.append(cos_sim[cos_sim_idx].item())
        index = sim.index(max(sim))
        torch.cuda.empty_cache()
        gc.collect()
        return names[index].split('.')[0]
