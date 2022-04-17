import cv2 as cv
from PIL import Image
import config
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

import torch

import os
from torch.utils.data import DataLoader,Dataset

class Kaiset(Dataset):
    def __init__(self,path, Listset = ["set00",'set01','set02','set06','set07']):
        self.path=path
        self.data=[]
        for sets in Listset:
            for v in os.listdir(self.path+'/'+sets):
                _tmp=os.listdir(self.path+'/'+sets+"/"+v+'/visible')
                _tmp=[self.path+'/'+sets+"/"+v+'/visible/'+x for x in _tmp]
                self.data.extend(_tmp)
        self.nbdata=len(self.data)
    def __getitem__(self, index):
        x=Image.open(self.data[index])
        x = config.transform(x)
        _tmp=""+self.data[index]
        _tmp=_tmp.replace('visible','lwir')
        y=Image.open(_tmp)
        y = config.transform(y)
        return x,y
    def __len__(self):
        return self.nbdata


if __name__ == "__main__":
    dataset = Kaiset(r'C:\Users\dell\Desktop\safe')
    loader = DataLoader(dataset, batch_size=6, shuffle= True)
    for x, y in loader:
        print(x.shape)
        save_image(x * 0.5 + 0.5 , "x.png")
        save_image(y * 0.5 + 0.5, "y.png")
        # save_image(x  , "x.png")
        # save_image(y , "y.png")
        import sys

        sys.exit()
