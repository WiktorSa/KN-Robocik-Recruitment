import numpy as np
import copy
import cv2
from typing import Tuple
from utils.GateEnum import GateEnum


class GateImage:
    def __init__(self, image: np.ndarray, image_width: int, image_height: int, center_x: int, center_y: int,
                 width: int, height: int):
        """
        GateImage class is designed to speed up the process of data preprocessing.

        :param image: image on which you can see the gate
        :param image_width: the width of the image
        :param image_height: the height of the image
        :param center_x: x coordinate of the center of the gate on the image
        :param center_y: y coordinate of the center of the gate on the image
        :param width: the width of the gate on the image
        :param height: the height of the gate on the image
        """

        self.image = image
        self.image_width = image_width
        self.image_height = image_height
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height

        self.gate_center = (center_x, center_y)
        # The coordinates of the top left corner of the gate
        self.top_left_corner = (int(center_x - width / 2), int(center_y - height / 2))
        # The coordinates of the bottom right corner of the gate
        self.bottom_right_corner = (int(center_x + width / 2), int(center_y + height / 2))
        self.gate_location = self.__get_gate_location()

    def __get_gate_location(self) -> int:
        """
        Get gate location from it's coordinates as a code (from GateEnum)

        :return: A code representing the gate location
        """

        if self.bottom_right_corner[0] > self.image_width:
            return GateEnum['right']
        elif self.top_left_corner[0] < 0:
            return GateEnum['left']
        elif self.bottom_right_corner[1] > self.image_height:
            return GateEnum['down']
        elif self.top_left_corner[1] < 0:
            return GateEnum['up']
        else:
            return GateEnum['fully_visible']

    def show_gate(self) -> None:
        """
        Function used primarily for testing.
        Using cv2 this function will show the following things:
        1. The image with the gate
        2. The center of the gate
        3. If possible it will show the location of the gate (red rectangle)
        4. It will print the location of the gate (see __get_gate_location())
        """

        image = cv2.circle(copy.deepcopy(self.image), self.gate_center, 10, (0, 0, 255), -1)
        if self.gate_location == GateEnum['fully_visible']:
            image = cv2.rectangle(image, self.top_left_corner, self.bottom_right_corner, (0, 0, 255), 2)

        # We want to have the descryption centered around x-axis
        text = "Gate location: " + self.gate_location.name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2

        textsize = cv2.getTextSize(text, font, font_scale, thickness)[0]
        location_text = ((self.image_width - textsize[0]) // 2, self.image_height - 30)

        image = cv2.putText(image, text, location_text, font, font_scale, (255, 255, 255), thickness)

        cv2.imshow("Gate Image", image)
        cv2.waitKey(0)

    def flip_image(self) -> 'GateImage':
        """
        Flip image across it's x-axis

        :return: GateImage object where image is flipped compared to the original image
        """

        new_image = cv2.flip(copy.deepcopy(self.image), 1)
        new_center_x = self.image_width - self.center_x

        return GateImage(new_image, self.image_width, self.image_height, new_center_x, self.center_y, self.width,
                         self.height)

    def get_image_data(self) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Return data about image needed later for training the models

        :return: tuple containing image, gate location and gate coordinates
        """

        gate_coordinates = np.array([self.top_left_corner[0], self.top_left_corner[1], self.bottom_right_corner[0],
                            self.bottom_right_corner[1]])

        return self.image.transpose((2, 0, 1)), self.gate_location.value, gate_coordinates
