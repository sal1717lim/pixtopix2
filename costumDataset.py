from PIL import Image
import config
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import random
import os
from torch.utils.data import DataLoader, Dataset
from median import MedianPool2d
from utils import SSIMLoss

#parsing and pre-processing the data
class Kaiset(Dataset):
    def __init__(self, path, Listset=["set00", 'set01', 'set02', 'set06', 'set07','set08'],train=True, shuffle=False):
        self.path = path
        self.data = []
        self.Listset=Listset[:-1] if train else Listset[-1:]
        for sets in self.Listset:
            for v in os.listdir(self.path + '/' + sets):
                _tmp = os.listdir(self.path + '/' + sets + "/" + v + '/visible')
                _tmp = [self.path + '/' + sets + "/" + v + '/visible/' + x for x in _tmp]
                self.data.extend(_tmp)
        self.nbdata = len(self.data)
        #if shuffle true, the data will be shuffeled before loading (used only in test data, in the trainign data is shuffeled using the loaded)
        if shuffle:
            for i in range(3):
                random.shuffle(self.data)

    def __getitem__(self, index):
        x = Image.open(self.data[index])
        x = config.transform(x)
        _tmp = "" + self.data[index]
        _tmp = _tmp.replace('visible', 'lwir')
        y = Image.open(_tmp)
        y = config.transform(y)
        return x, y

    def __len__(self):
        return self.nbdata

class Kaisetdepth(Dataset):
    def __init__(self, path, depthPath, Listset=["set00", 'set01', 'set02', 'set06', 'set07','set08'],train=True, shuffle=False):
        self.path = path
        self.depthpath = depthPath
        self.data = []
        self.depth = []
        self.Listset=Listset[:-1] if train else Listset[-1:]
        for sets in self.Listset:
            for v in os.listdir(self.path + '/' + sets):
                _tmp = os.listdir(self.path + '/' + sets + "/" + v + '/visible')
                _tmp2 = os.listdir(self.path + '/' + sets + "/" + v + '/visible')
                _tmp = [self.path + '/' + sets + "/" + v + '/visible/' + x for x in _tmp]
                _tmp2 = [self.depthpath + '/' + sets + "/" + v + '/' + x for x in _tmp2]
                self.data.extend(_tmp)
                self.depth.extend(_tmp2)
        self.nbdata = len(self.data)
        #if shuffle true, the data will be shuffeled before loading (used only in test data, in the trainign data is shuffeled using the loaded)
        if shuffle:
            for i in range(3):
                random.shuffle(self.data)

    def __getitem__(self, index):
        x = Image.open(self.data[index])
        x2 =Image.open(self.depth[index])
        x = config.transform(x)
        x2 =config.transform(x2)
        _tmp = "" + self.data[index]
        _tmp = _tmp.replace('visible', 'lwir')
        y = Image.open(_tmp)
        y = config.transform(y)
        return x,x2, y

    def __len__(self):
        return self.nbdata
#testing if everything works proprely
if __name__ == "__main__":
    dataset = Kaiset(config.TRAIN_DIR, Listset=config.TRAIN_LIST)
    loader = DataLoader(dataset, batch_size=6, shuffle=True)
    for x, y in loader:
        print(x.shape)
        print()
        save_image(x * 0.5 + 0.5, "x.png")
        save_image(MedianPool2d()(x) * 0.5 + 0.5, "median.png")
        save_image(y * 0.5 + 0.5, "y.png")
        # save_image(x  , "x.png")
        # save_image(y , "y.png")
        import sys

        sys.exit()
