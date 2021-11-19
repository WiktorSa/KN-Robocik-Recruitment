import random
from typing import List
from preprocessing.GateImage import GateImage


def perform_flip_augmentation(gate_images: List[GateImage], chance_flip: float = 1.0) -> List[GateImage]:
    """
    Perform flip augmentation on the images

    :param gate_images: gate images on which we want to perform flip augmentation
    :param chance_flip: chance for the image to be flipped
    :return: list of flipped GateImage objects
    """

    flipped_gate_images = []
    for gate_image in gate_images:
        if random.random() < chance_flip:
            flipped_gate_images.append(gate_image.flip_image())

    return flipped_gate_images


def perform_crop_augmentation(gate_images: List[GateImage], chance_crop: float = 0.75) -> List[GateImage]:
    """
    Perform crop augmentation on the images

    :param gate_images: gate images on which we want to perform crop augmentation
    :param chance_crop: chance for the image to be cropped
    :return: list of cropped GateImage objects
    """

    cropped_gate_images = []
    for gate_image in gate_images:
        if random.random() < chance_crop:
            cropped_gate_images.append(gate_image.crop(random.randint(25, 250), random.randint(25, 250),
                                                       random.randint(25, 150), random.randint(25, 150)))

    return cropped_gate_images


def perform_color_jitter_augmentation(gate_images: List[GateImage], chance_color_jitter: float = 0.75) -> \
        List[GateImage]:
    """
    Perform color jitter augmentation on the images

    :param gate_images: gate images on which we want to perform color jitter augmentation
    :param chance_color_jitter: chance for the image to be color jittered
    :return: list of color jitted GateImage objects
    """

    color_jitted_gate_images = []
    for gate_image in gate_images:
        if random.random() < chance_color_jitter:
            color_jitted_gate_images.append(gate_image.color_jitter_image(
                random.uniform(0, 1), random.uniform(0, 1.25), random.uniform(0, 1.25), random.uniform(0, 0.1)
            ))

    return color_jitted_gate_images
