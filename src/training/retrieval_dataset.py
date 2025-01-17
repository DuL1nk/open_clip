import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import json


class GenericDataset(data.Dataset):
    """
    Dataset loader for all datasets.
    """

    def __init__(self, transform, split, data_name, root_path):
        self.root_path = root_path
        self.transform = transform
        self.split = split
        self.data_name = data_name


        self.img_path = root_path

        img_file_path = os.path.join(root_path, "{}_{}_ims.txt".format(self.data_name, self.split))
        with open(img_file_path, "r") as fp:
            self.images = fp.readlines()
        self.images = [image.strip() for image in self.images]

        cap_file_path = os.path.join(root_path, "{}_{}_caps.txt".format(self.data_name, self.split))
        with open(cap_file_path, "r") as fp:
            self.captions = fp.readlines()
        self.captions = [caption.strip() for caption in self.captions]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        image_name = self.images[index]
        caption = self.captions[index]

        if self.data_name == 'wiki':
            max_context_len = 20
        else:
            max_context_len = 60

        if self.data_name == 'f30k' or self.data_name == 'wiki':
            img_path = self.img_path
        else:
            coco_split_folder = image_name.split("_")[1]
            img_path = os.path.join(self.img_path, coco_split_folder)

        short_caption = caption.split(' ')
        short_caption = short_caption[:max_context_len]
        caption = ' '.join(short_caption)

        image = self.transform(Image.open(os.path.join(img_path, image_name)))

        return image, caption, index, image_name

    def __len__(self):

        return len(self.images)

        if self.split == 'test' and self.data_name == 'mscoco':  # TO CHECK
            return int(len(self.images) / 5)
        else:
            return len(self.images)


#################

def get_loader(transform, split, data_name, data_root_path, batch_size, num_workers):
    """Returns torch.utils.data.DataLoader for custom dataset."""

    dataset = GenericDataset(transform, split, data_name, data_root_path)
    # Data loader
    if split == 'train':
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=num_workers)

    else:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=num_workers)
    return data_loader


def get_split_loader(split, data_name, data_root_path, batch_size, workers, preprocess):
    # Build Dataset Loader
    transform = preprocess

    loader = get_loader(transform, split, data_name, data_root_path, batch_size, workers)

    return loader

