# this file was added completely new

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
#from sklearn.model_selection import train_test_split
from .decoders import ImageDataDecoder
import random
#import numpy as np
#from PIL import Image

# these functions were just created to check how the augmented images/crops look like
def save_image(tensor, name1, name2):
    # Convert the tensor to a NumPy array
    #numpy_array = tensor.cpu().numpy()
    
    # Convert to uint8 and scale to [0, 255]
    #numpy_array = numpy_array
    
    # Create an image from the NumPy array
    #image = Image.fromarray(numpy_array.transpose(1, 2, 0))  # Convert to HWC format
    
    # Concatenate name1 and name2 to form the file name
    file_name = f'{name1}_{name2}.png'
    
    # Specify the save path with the concatenated file name
    save_path = f'/home/ubuntu/example_image/augmented_images/{file_name}'
    
    # Save the image
    tensor.save(save_path)

def save_all(image_pil, name1):
    # save both global crops
    save_image(image_pil['global_crops'][0], name1, 'global_crop1')
    save_image(image_pil['global_crops'][1], name1, 'global_crop2')

    #save local crops
    local_crops = image_pil['local_crops']
    for i, image in enumerate(local_crops):
        save_image(image, name1, 'local_crop'+str(i))


class CustomImageDataset(Dataset):
    def __init__(
            self,
            split,
            root: str,
            transform=None,
            target_transform=None,
            test_size=0.2,
            random_state=42):
        self.img_dir = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        # split into train and test
        #self.train_data, self.test_data = train_test_split(
        #    self.img_labels, test_size=test_size, random_state=random_state)

    def __len__(self):
        #return len(self.img_labels) 
        return len([f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f))])

    def __getitem__(self, idx):
        try:
            img_path = [f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f))][idx]
            img_path = os.path.join(self.img_dir, img_path)
            with open(img_path, mode="rb") as f:
                image_pil = f.read()
            image_pil = ImageDataDecoder(image_pil).decode()
        except Exception as e:
            # in case of error when reading image, just take a random different one
            random_index = random.randint(0, len(self) - 1)
            image = self.__getitem__(random_index)
            return image, None
        if self.transform:
            image_pil = self.transform(image_pil)

        return image_pil, None

    def get_test_item(self, idx):
        img_path = os.path.join(self.img_dir, self.test_data.iloc[idx, 0])
        image = read_image(img_path)
        image_pil = transforms.ToPILImage()(image)
        if self.transform:
            image_pil = self.transform(image_pil)
        return image_pil
    
