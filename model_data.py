
'''
File that contains important classes of data models, prediction models, loss etc
'''


import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoProcessor
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader


class PriceDataset(Dataset):
    def __init__(self, annotations_file, image_dir, content_col, return_target=True, use_log_scale=True):
        self.annotations = pd.read_csv(annotations_file)
        self.image_dir = image_dir
        self.content_col = content_col
        self.return_target = return_target
        self.use_log_scale = use_log_scale
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.loc[idx]
        name = Path(row['image_link']).name
        try:
            img = Image.open(f'{self.image_dir}/{name}').convert('RGB')
        except:
            img = Image.new('RGB', (300, 300), color=(0, 0, 0))
        text = row[self.content_col]
        price = row['price'] if self.return_target else 0
        return img, text, np.log1p(price) if self.use_log_scale else price



def collate_function(batch):
    batch = list(map(list, zip(*batch)))
    batch[-1] = torch.tensor(batch[-1], dtype=torch.float)
    return batch

class SMAPELoss(torch.nn.Module):
    def __init__(self, epsilon=1e-3, log_smooth=True):
        super(SMAPELoss, self).__init__()
        self.epsilon = epsilon
        self.log_smooth = log_smooth

    def forward(self, predictions, targets):
        # diff = torch.abs(predictions - targets)
        # denom = torch.sqrt(predictions**2 + targets**2 + self.epsilon)
        # ratio = 2.0 * diff / denom
        # if self.log_smooth:
        #     ratio = torch.log(ratio)  # smooth large gradients
        # return ratio.mean()
        return (
                torch.abs(predictions-targets) / ((torch.abs(predictions) + torch.abs(targets))/2)
        ).mean()

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, predictions, targets):
        return torch.log(torch.cosh(predictions-targets)).mean()



class PriceModel(torch.nn.Module):
    def __init__(self, checkpoint, cache_dir = None):
        super(PriceModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(checkpoint, device_map='auto', cache_dir = cache_dir)
        self.processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=cache_dir)
        self.head = torch.nn.Sequential(torch.nn.Linear(2048, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 512), torch.nn.ReLU(), 
                                        torch.nn.Linear(512, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)).to(device=self.backbone.device)

    def forward(self, images, texts):
        inputs = self.processor(images=images, text=texts, return_tensors="pt", truncation=True, padding="max_length").to(self.backbone.device)
        image_features = self.backbone.get_image_features(pixel_values = inputs['pixel_values'])
        text_features = self.backbone.get_text_features(input_ids = inputs['input_ids'])
        preds = self.head(torch.cat([image_features, text_features], dim=-1))
        return preds.squeeze()

    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint, weights_only=False)
        self.load_state_dict(checkpoint['model'])



