import random
from typing import List
from preprocessing.GateImage import GateImage


def perform_flip_augmentation(gate_images: List[GateImage]) -> List[GateImage]:
    """
    Perform flip augmentation on some of the images

    :param gate_images: gate images on which we want to perform flip augmentation
    :return: list of flipped GateImage objects
    """

    CHANCE_FOR_FLIP_AUGMENTATION = 0.9

    flipped_gate_images = []
    for gate_image in gate_images:
        if random.random() < CHANCE_FOR_FLIP_AUGMENTATION:
            flipped_gate_images.append(gate_image.flip_image())

    return flipped_gate_images


def perform_center_crop_augmentation(gate_images: List[GateImage]) -> List[GateImage]:
    """
    Perform center crop augmentation on some of the images

    :param gate_images: gate images on which we want to perform center crop augmentation
    :return: list of center cropped GateImage objects
    """

    CHANCE_FOR_CENTER_CROP_AUGMENTATION = 0.8

    center_cropped_gate_images = []
    for gate_image in gate_images:
        if random.random() < CHANCE_FOR_CENTER_CROP_AUGMENTATION:
            center_cropped_gate_images.append(gate_image.center_crop(random.randint(25, 175), random.randint(25, 175),
                                                              random.randint(25, 100), random.randint(25, 100)))

    return center_cropped_gate_images


def perform_color_jitter_augmentation(gate_images: List[GateImage]) -> List[GateImage]:
    """
    Perform color jitter augmentation on some of the images

    :param gate_images: gate images on which we want to perform color jitter augmentation
    :return: list of color jitted GateImage objects
    """

    CHANCE_FOR_COLOR_JITTER_AUGMENTATION = 0.7

    color_jitted_gate_images = []
    for gate_image in gate_images:
        if random.random() < CHANCE_FOR_COLOR_JITTER_AUGMENTATION:
            color_jitted_gate_images.append(gate_image.color_jitter_image(
                random.uniform(0, 1), random.uniform(0, 1.25), random.uniform(0, 1.25), random.uniform(0, 0.1)
            ))

    return color_jitted_gate_images
