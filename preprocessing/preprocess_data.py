import copy
import numpy as np
import itertools
from os import mkdir
from os.path import join, isdir
from typing import Tuple, List
from preprocessing.load_data import load_data
from preprocessing.gate_image import GateImage
from preprocessing.data_augmentation import perform_flip_augmentation, perform_crop_augmentation, \
    perform_color_jitter_augmentation


def preprocess_data(directory: str, train_size: float, val_size: float, use_augmentation: bool,
                    save_directory: str, seed: int) -> None:
    """
    Preprocess data and save it later in a given folder.
    The preprocessing includes:
    1. Loading data from a given directory
    2. Dividing data into training, validation and test set
    3. Using data augmentation techniques on training set if needed
    4. Saving data in the .npz format in a given directory

    :param directory: directory where images and data about gates are stored
    :param train_size: size of training data
    :param val_size: size of validation data
    :param use_augmentation: should we augmentate data
    :param save_directory: directory where preprocessed data should be saved
    :param seed: seed
    """

    gate_images = load_data(directory)

    rng = np.random.default_rng(seed)
    rng.shuffle(gate_images)

    train_index = int(len(gate_images) * train_size)
    val_index = int(len(gate_images) * (train_size + val_size))

    # We separate data so that we don't perform data augmentation techniques on validation or test set
    train_gate_images = gate_images[:train_index]
    val_gate_images = gate_images[train_index:val_index]
    test_gate_images = gate_images[val_index:]

    # Use augmentation techniques to increase the number of training data
    if use_augmentation:
        train_gate_images.extend(perform_flip_augmentation(train_gate_images))
        train_gate_images.extend(perform_crop_augmentation(train_gate_images))
        train_gate_images.extend(perform_color_jitter_augmentation(train_gate_images))

        # Shuffle all images created by data augmentation
        rng.shuffle(train_gate_images)

    train_images, train_gate_locations, train_gate_coordinates = get_model_input(train_gate_images)
    val_images, val_gate_locations, val_gate_coordinates = get_model_input(val_gate_images)
    test_images, test_gate_locations, test_gate_coordinates = get_model_input(test_gate_images)

    # Create a directory if needed
    if not isdir(save_directory):
        mkdir(save_directory)

    np.savez_compressed(join(save_directory, 'train_data.npz'), images=train_images,
                        gate_locations=train_gate_locations, gate_coordinates=train_gate_coordinates)
    np.savez_compressed(join(save_directory, 'val_data.npz'), images=val_images,
                        gate_locations=val_gate_locations, gate_coordinates=val_gate_coordinates)
    np.savez_compressed(join(save_directory, 'test_data.npz'), images=test_images,
                        gate_locations=test_gate_locations, gate_coordinates=test_gate_coordinates)


def get_model_input(gate_images: List[GateImage]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get data from GateImage objects that will later be used for training the models
    Warning - images will be reshape to shape 200 x 125 so that they could form an input to the model
    
    :param gate_images: list of GateImage objects
    :return: Three numpy arrays containing: images, gate locations and gate coordinates
    """

    NEW_WIDTH = 200
    NEW_HEIGHT = 125

    # Reshape all images so that the input to the model will be smaller
    for gate_image in gate_images:
        gate_image.reshape(NEW_WIDTH, NEW_HEIGHT)

    no_images = len(gate_images)
    # There are 3 color channels
    # And there are 4 coordinates to predict
    images = np.empty(shape=(no_images, 3, NEW_HEIGHT, NEW_WIDTH), dtype=np.int)
    gate_locations = np.empty(shape=no_images, dtype=np.int)
    gate_coordinates = np.empty(shape=(no_images, 4), dtype=np.int)

    for i in range(no_images):
        images[i], gate_locations[i], gate_coordinates[i] = gate_images[i].get_image_data()

    return images, gate_locations, gate_coordinates
