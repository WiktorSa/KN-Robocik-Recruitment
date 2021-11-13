import numpy as np
import itertools
from os import mkdir
from os.path import join, isdir
from preprocessing.LoadData import load_data


def preprocess_data(directory: str, train_size: float, val_size: float, use_flip_augmentation: bool,
                    save_folder: str, seed: int) -> None:
    """
    Preprocess data and save it later in a given folder.
    The preprocessing includes:
    1. Loading data from a given directory
    2. Dividing data into training, validation and test set
    3. Using data augmentation techniques on training set if needed
    4. Saving data in .npz format in a given directory

    :param directory: directory where images and data about gates are stored
    :param train_size: size of training data
    :param val_size: size of validation data
    :param use_flip_augmentation: should flip be used as a data augmentation technique
    :param save_folder: folder where preprocessed data should be saved
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

    flipped_gate_images = []
    if use_flip_augmentation:
        flipped_gate_images = [gate_image.flip_image() for gate_image in train_gate_images]

    all_train_gate_images = list(itertools.chain(train_gate_images, flipped_gate_images))

    train_images, train_gate_location, train_gate_coordinates = None, None, None

    print(len(train_gate_images))
    print(len(all_train_gate_images))
