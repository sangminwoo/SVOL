import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
from torchvision import io

class FrameDataset(Dataset): 
    def __init__(self, root: str, frame: int = 64):
        super(FrameDataset).__init__()
        img_path = Path(root)
        self.files = list(img_path.rglob('*.JPEG'))
        self.files.sort()
        # self.files = self.files[:(len(self.files)//frame)*frame:((len(self.files)//frame))]
        
        if len(self.files) >= frame:
            stride = len(self.files)/frame
            self.files = [self.files[round(i*stride)] for i in range(frame)]
    
    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx): 
        img_path = str(self.files[idx])
        image = io.read_image(img_path)
        return image

if __name__ == '__main__':
    root = '/mnt/server12_hard3/sangmin/data/svol/imagenet_vid/Data/VID/train/ILSVRC2015_train_00000000'
    frame = 64
    dataset = FrameDataset(root, frame=frame)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=frame)
    image = next(iter(dataloader))
    print(image.shape)