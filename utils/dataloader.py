import torch
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from utils.processor import Samprocessor


class DatasetSegmentation(Dataset):
    """
    Dataset to process the images and masks

    Arguments:
        folder_path (str): The path of the folder containing the images
        processor (obj): Samprocessor class that helps pre processing the image, and prompt 
    
    Return:
        (dict): Dictionnary with 4 keys (image, original_size, boxes, ground_truth_mask)
            image: image pre processed to 1024x1024 size
            original_size: Original size of the image before pre processing
            boxes: bouding box after adapting the coordinates of the pre processed image
            ground_truth_mask: Ground truth mask
    """

    def __init__(self, config_file: dict, processor: Samprocessor, mode: str):
        super().__init__()
        if mode == "train":
            path = config_file["DATASET"]["PATH"]
            image_path = path + "/images/"
            mask_path = path + "/masks/"
            self.img_files = [image_path + f for f in listdir(image_path) if isfile(join(image_path, f))]
            self.mask_files = [mask_path + f for f in listdir(mask_path) if isfile(join(mask_path, f))]

        self.processor = processor

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index: int) -> list:
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            # get image and mask in PIL format
            image =  Image.open(img_path)
            mask = Image.open(mask_path)
            image = image.convert("RGB")
            mask = mask.convert('1')
            ground_truth_mask =  np.array(mask)
            original_size = tuple(image.size)[::-1]
    
            # get bounding box prompt
            box = get_bounding_box(ground_truth_mask)
            inputs = self.processor(image, original_size, box)
            inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask)

            return inputs
    
def collate_fn(batch: torch.utils.data) -> list:
    """
    Used to get a list of dict as output when using a dataloader

    Arguments:
        batch: The batched dataset
    
    Return:
        (list): list of batched dataset so a list(dict)
    """
    return list(batch)

def get_bounding_box(ground_truth_map: np.array) -> list:
    """
    Get the bounding box of the image with the ground truth mask

    Arguments:
        ground_truth_map: Take ground truth mask in array format

    Return:
        bbox: Bounding box of the mask [X, Y, X, Y]

    """
    # get bounding box from mask
    idx = np.where(ground_truth_map > 0)
    x_indices = idx[1]
    y_indices = idx[0]
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox