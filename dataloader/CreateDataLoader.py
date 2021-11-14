import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple
from utils.GateEnum import GateEnum
from dataloader.GateClassificationDataset import GateClassificationDataset
from dataloader.GateRegressionDataset import GateRegressionDataset


def get_data_loader(images: np.ndarray, gate_locations: np.ndarray, gate_coordinates: np.ndarray, batch_size: int,
                    is_classification_task: bool) -> DataLoader:
    """
    Create dataloader for a given problem

    :param images: images containing gates
    :param gate_locations: codes representing the location of the gate
    :param gate_coordinates: the coordinates of the gate
    :param batch_size: batch size
    :param is_classification_task: are we creating dataloader for classification task.
    If True return DataLoader for classification task, else return DataLoader for regression task
    :return: DataLoader appriopriate for a given task
    """

    if is_classification_task:
        dataset = GateClassificationDataset(images, gate_locations)

    else:
        images_fully_visible, gate_coordinates_fully_visible = get_only_visible_gates(images, gate_locations,
                                                                                      gate_coordinates)
        dataset = GateRegressionDataset(images_fully_visible, gate_coordinates_fully_visible)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_only_visible_gates(images, gate_locations, gate_coordinates) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get only visible gates (regression model can only learn on fully visible gates)

    :param images: images
    :param gate_locations: codes representing the location of the gate
    :param gate_coordinates: the coordinates of the gate
    :return: images and gate coordinates of gates that are fully visible
    """

    images_fully_visible = []
    gate_coordinates_fully_visible = []
    for i in range(len(images)):
        if gate_locations[i] == GateEnum['fully_visible'].value:
            images_fully_visible.append(images[i])
            gate_coordinates_fully_visible.append(gate_coordinates[i])

    return np.array(images_fully_visible), np.array(gate_coordinates_fully_visible)
