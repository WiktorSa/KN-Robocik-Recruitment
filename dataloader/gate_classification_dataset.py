import numpy as np
import torch
from torch.utils.data import Dataset


class GateClassificationDataset(Dataset):
    def __init__(self, images: np.ndarray, gate_locations: np.ndarray):
        """
        Create a gate location dataset that will be used for classification task

        :param images: images that contain gates
        :param gate_locations: codes representing the location of gates
        """

        self.images = images
        self.gate_locations = gate_locations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # We divide by 255.0 to have values in the image between 0 and 1
        return self.images[idx] / 255.0, self.gate_locations[idx]
