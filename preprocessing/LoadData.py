import cv2
import os
import glob
from typing import List
from preprocessing.GateImage import GateImage


def load_data(directory: str) -> List[GateImage]:
    """
    Read images and data about gates from a given directory

    :param directory: the directory where files are located
    :return: list containing all gate images and their info (GateImage class)
    """

    gate_images = []

    for file in glob.glob(os.path.join(directory, '*.jpg')):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        gate_info = read_gate_info(file.replace('jpg', 'txt'), image_width, image_height)
        gate_image = GateImage(image, image_width, image_height, gate_info[1], gate_info[2], gate_info[3], gate_info[4])
        gate_images.append(gate_image)

    return gate_images


def read_gate_info(file: str, image_width: int, image_height: int) -> List[int]:
    """
    Read info about the gate from a given file. Then convert it using width and height of the appropriate image
    so that the info about the gate corresponds to the image

    :param file: the location of the file
    :param image_width: the width of the image
    :param image_height: the height of the image
    :return: the list containing info about the gate (detected, center_x, center_y, width, height in given order)
    """

    with open(file) as f:
        info_gate = f.readline().split()

    # readline will read everything as string. We need to convert the values to float
    info_gate = list(map(float, info_gate))

    # Convert to proper values
    info_gate[1] *= image_width
    info_gate[2] *= image_height
    info_gate[3] *= image_width
    info_gate[4] *= image_height

    # To make our job easier we will convert everything to int
    info_gate = list(map(int, info_gate))

    return info_gate
