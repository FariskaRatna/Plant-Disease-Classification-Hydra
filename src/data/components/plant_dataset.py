from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import Dataset

class PlantDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))

    def __len__(self):
        return sum(len(files) for _, _, files in os.walk(self.data_dir))
    
    def __getitem__(self, idx):
        label = 0
        for i, c in enumerate(self.classes):
            if idx < len(os.listdir(os.path.join(self.data_dir, c))):
                label = i
                break
            else:
                idx -= len(os.listdir(os.path.join(self.data_dir, c)))
        
        image_name = os.listdir(os.path.join(self.data_dir, self.classes[label]))[idx]
        image = Image.open(os.path.join(self.data_dir, self.classes[label], image_name))
        if self.transform:
            image = self.transform(image)

        return image, label