from torch.utils.data import Dataset
from PIL import Image
import os

# Dataset
class ChestXrayDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, sensitive_attr='gender'):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.sensitive_attr = sensitive_attr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image'])
        image = Image.open(img_path).convert('RGB')
        label = int(row['class'])
        sensitive = int(row[self.sensitive_attr])
        if self.transform:
            image = self.transform(image)
        return image, label, sensitive
