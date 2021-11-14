import numpy as np
import torch
from torch.utils.data import Dataset


class GateRegressionDataset(Dataset):
    def __init__(self, images: np.ndarray, gate_coordinates: np.ndarray):
        """
        Create a gate location dataset that will be later used for regression
        :param images: images that contain gates saved as numpy arrays
        :param gate_coordinates: coordinates representing the location of the gate
        (top_left_corner and bottom_right_corner)
        """

        self.images = images
        self.gate_coordinates = gate_coordinates

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # We divide by 255.0 to have values in images between 0 and 1
        return self.images[idx] / 255.0, self.gate_coordinates[idx]
