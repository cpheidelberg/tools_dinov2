import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.models import resnet18 as CN_module
from . import FSL_network 
import h5py
import random
import timm

class Dataset_torch(Dataset):
    def __init__(self, img_pngs, img_labels, transform=None):
        self.img_labels = torch.from_numpy(img_labels)
        self.img_pngs = torch.from_numpy(img_pngs).to(torch.float)
        self.transform = transform


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.img_pngs[idx]
        label = self.img_labels[idx]


        if self.transform:
         #   image *= 255
         #   image = image.to(torch.uint8)
            image = self.transform(image)
          #  image = image.to(torch.float)/ 255

        return image, label

    def get_labels(self):
        return self.img_labels.numpy()

class CustomImageDataset(Dataset):
    def __init__(
            self,
            root: str,
            transform=None,
            target_transform=None,
            test_size=0.2,
            random_state=42,
            check_input = False):
        self.img_dir = root
        self.transform = transform
        self.target_transform = target_transform
        if not self.img_dir.endswith(".h5"):
            self.file_list = [f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f)) and not f.startswith('.')]
        else:
            self.file_list = "its a h5 file"
        #len(self.file_list)
        if check_input and not self.img_dir.endswith(".h5"):
            self.file_list = check_file_path(self.file_list)
        #len(self.file_list)
        #print(self.file_list)
        # split into train and test
        #self.train_data, self.test_data = train_test_split(
        #    self.img_labels, test_size=test_size, random_state=random_state)

    def __len__(self):
        #return len(self.img_labels)
        if not self.img_dir.endswith(".h5"):
            length = len(self.file_list)
        elif self.img_dir.endswith(".h5"):
            with h5py.File(self.img_dir, mode='r') as h5f:
                dst = h5f['tile']
                length = dst.shape[0]

        return  length #len([f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f))])

    def __getitem__(self, idx, verbose = False):

        if os.path.isdir(self.img_dir):

            img_path = self.file_list[idx]
            img_path = os.path.join(self.img_dir, img_path)
            #print("does work until here")
            if 'vips' not in img_path:
                with open(img_path, mode="rb") as f:
                    image_pil = f.read()
                image_pil = ImageDataDecoder(image_pil).decode()
                if verbose:
                    print(f"does work until here; and uses vips input; index is {idx}")
            else:
                #print(img_path)

                vips_image = pyvips.Image.new_from_file(img_path)
                numpy_image = np.array(vips_image)
                image_pil = Image.fromarray(numpy_image)#.tobytes()
                if verbose:
                    print(f"does work until here; and uses vips input; index is {idx}")

        if self.img_dir.endswith(".h5"):

            with h5py.File(self.img_dir, mode='r') as h5f:
                dst = h5f['tile']
                numpy_image = np.array(dst[idx]).astype("uint8")
                numpy_image = (numpy_image/255 - mean / std) 
                numpy_image = np.transpose(numpy_image, axes=(2, 1, 0,)) 

           #     image_pil = Image.fromarray(numpy_image)

            if verbose:
                print(f"it worked until here; and used h5 input; input is {idx}")


        if self.transform:
            image_pil = self.transform(image_pil)

        return torch.from_numpy(numpy_image).to(torch.float)



mean = np.array([0.485, 0.456, 0.406], dtype = np.float32)  # Mean values for RGB channels
std = np.array([0.229, 0.224, 0.225], dtype = np.float32)  # Std values for RGB channels



def load_images_from_folder(folder, file_format = '.tif', target_size=(200, 200), num_to_load=None):
    images = []

    # Get list of all image files in the folder
    image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(file_format )]

    if num_to_load > len(image_files):
        num_to_load =  len(image_files)       

    print(len(image_files))
    selected_files = random.sample(image_files, num_to_load)
    print(len(selected_files))


    # Load selected images
    for filename in selected_files:
        try:
            img = Image.open(os.path.join(folder, filename))
        except:
            continue
        if target_size is not None:
            img = img.resize(target_size)  

        img = np.array(img, dtype= np.float32)[:,:,:3]/255       
        img = img - mean/std
        if np.any(img):
            images.append(img)

    images = np.asarray(images)
    try:
        return images.transpose(0, 3, 1, 2)
    except:
        return None

import os
import numpy as np
from sklearn.utils import shuffle as sklearn_shuffle

def load_images_from_directories(paths, target_size=(448, 448), file_format = '.tif', min_images=1, shuffle=False, normalize = True, mode = 'png'):

    if mode == 'png':
        images = None    
        images_classlabels = None

        for index, path in enumerate(paths):
            # Get the number of images in the directory
            all_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            num_images = len(all_files)
            
            if num_images == 0:
                continue

            # Determine the number of images to load based on percentage and min_images
            num_to_load = min_images


            # Load the specified number of images
            selected_images = load_images_from_folder(path, target_size=target_size, num_to_load=num_to_load)

            if selected_images is None or len(selected_images) == 0:
                continue

            if images is None:
                images = selected_images
                images_classlabels = np.full(len(selected_images), index, dtype=np.int8)
            else:
                images = np.vstack((images, selected_images))
                images_classlabels = np.append(images_classlabels, np.full(len(selected_images), index, dtype=np.int8))
            
            print(f"Loaded {len(selected_images)} images from directory {path}")

        if shuffle and images is not None and images_classlabels is not None:
            images, images_classlabels = sklearn_shuffle(images, images_classlabels)

    if mode == 'hdf':
        # Random indices to extract images
        img_idx = np.random.randint(10, 2000000, size=3000)

        # Open the HDF5 file
        with h5py.File(paths, mode='r') as h5f:

            # Access the dataset
            dst = h5f['tile']
            
            # Extract images for the indices in img_idx
            images = np.asarray([np.array(dst[idx]).astype("uint8")/255 - mean/std for idx in img_idx])
            images = np.transpose(images, axes=(0, 3, 2, 1))
            images_classlabels = np.zeros(len(images))

    return images, images_classlabels



def get_dino_backbone(dict_path, device):
    """
    Load the DINO backbone model (teacher model) and load the correct state dict

    Args:
    dict_path (str): Path to the dictionary containing the pretrained weights.
    device (str): Device on which to map the model ('cpu' or 'cuda').

    Returns:
    model_teacher (torch.nn.Module): The teacher model loaded with corrected weights.
    """

    pretrained = torch.load(dict_path, map_location=torch.device(device))
    if 'model' in  pretrained:
    # Load the pretrained weights from the provided checkpoint
        pretrained = torch.load(dict_path, map_location=torch.device(device))['model']
        teacher_state_dict = {k.replace('teacher.', ''): v for k, v in pretrained.items() if k.startswith('teacher.')}
    else:
        teacher_state_dict = pretrained['teacher']

    # Prepare teacher's state dict for loading by removing 'backbone.' prefix
    teacher_state_dict_corrected = {}
    for key, value in teacher_state_dict.items():
        if 'dino_head' in key:
            print('dino_head not used')  # Skipping the classification head
        else:
            new_key = key.replace('backbone.', '')  # Remove 'backbone.' from keys
            teacher_state_dict_corrected[new_key] = value

    model_teacher = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
    
    model_teacher.load_state_dict(teacher_state_dict_corrected, strict=False)

    # Return teacher model
    return model_teacher.to(device)




def load_model_pytorch(backbone_path , device = 'cuda'):
        
    
    if backbone_path:
        model = get_dino_backbone(backbone_path, device)
        model.eval()
    else:    
        print('No checkpoint, load Uni weights...')
        import timm
        model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
    
        model.load_state_dict(torch.load("/home/na236/pytorch_model_uni.bin"), strict=False)
        model.eval

    model_classifier = FSL_network.PrototypicalNetworks(model)  
    model_classifier.eval()

    return model_classifier.to(device)  





