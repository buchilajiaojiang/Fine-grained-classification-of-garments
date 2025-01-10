from torch.utils.data import Dataset, DataLoader
import os
import json
from torchvision import transforms
from PIL import Image


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "./"))


def json_reader(label_path):
    # 打开JSON文件，逐行读取并解析
    # label = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line)
            # print(json_obj)
            # label.append(json_obj)
    # return label
    return json_obj


class BaseDataset(Dataset):
    def __init__(self, num_classes, data_type, img_size, dataset_name):
        self.img_path = os.path.join(root_dir, 'data', dataset_name, data_type, "image")
        self.label_path = os.path.join(root_dir, 'data', dataset_name, data_type, "label", "label.json")
        self.labels = json_reader(self.label_path)
        self.len = len(self.labels)
        self.num_classes = num_classes
        
        rate = 0.875

        if data_type == "train":
            self.transform = transforms.Compose([
                transforms.Resize((int(img_size // rate), int(img_size // rate))),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=32.0 / 255.0, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((int(img_size // 0.875), int(img_size // 0.875))),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        labels = self.labels[idx]
        image_path = os.path.join(self.img_path, labels[0])
        image = Image.open(image_path)
        image = self.transform(image)
        return image, labels[1]

    def __len__(self):
        return self.len