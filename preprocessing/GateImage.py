import numpy as np
import copy
import cv2
from PIL import Image
from torchvision.transforms import ColorJitter
from typing import Tuple, List
from utils.GateEnum import GateEnum


class GateImage:
    def __init__(self, image: np.ndarray, image_width: int, image_height: int, center_x: int, center_y: int,
                 width: int, height: int):
        """
        Create GateImage class that is designed to speed up the process of data preprocessing and data augmentation

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

    def __get_gate_location(self) -> GateEnum:
        """
        Get gate location from it's coordinates as an enum (GateEnum)

        :return: An enum representing the gate location
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

    # This function is necessary. Otherwise the input for our model will be too large
    def reshape(self, image_width, image_height) -> None:
        """
        Reshape image and change all data so that it matches to the reshaped image

        :param image_width: new width of a image
        :param image_height: new height of a image
        """

        proportion_x = image_width / self.image_width
        propotion_y = image_height / self.image_height

        self.image = cv2.resize(self.image, (image_width, image_height), interpolation=cv2.INTER_AREA)
        self.image_width = image_width
        self.image_height = image_height
        self.center_x = int(self.center_x * proportion_x)
        self.center_y = int(self.center_y * propotion_y)
        self.width = int(self.width * proportion_x)
        self.height = int(self.height * propotion_y)
        self.gate_center = (self.center_x, self.center_y)
        self.top_left_corner = (int(self.center_x - self.width / 2), int(self.center_y - self.height / 2))
        self.bottom_right_corner = (int(self.center_x + self.width / 2), int(self.center_y + self.height / 2))

    def flip_image(self) -> 'GateImage':
        """
        Flip image across it's x-axis

        :return: GateImage object where image is flipped compared to the original image
        """

        new_image = cv2.flip(copy.deepcopy(self.image), 1)
        new_center_x = self.image_width - self.center_x

        return GateImage(new_image, self.image_width, self.image_height, new_center_x, self.center_y, self.width,
                         self.height)

    def color_jitter_image(self, brightness: float = 0, contrast: float = 0, saturation: float = 0,
                           hue: float = 0) -> 'GateImage':
        """
        Perform color jittering based on a given parameters

        :param brightness: how much to jitter brightness
        :param contrast how much to jitter contrast
        :param saturation: how much to jitter saturation
        :param hue: how much to jitter hue
        :return: GateImage object with image on which color jittering was performed
        """

        # ColorJitter only accepts PIL images so we need to convert to PIL image and later convert back to numpy array
        transformer = ColorJitter(brightness, contrast, saturation, hue)

        transformed_image = Image.fromarray(self.image)
        transformed_image = transformer(transformed_image)
        transformed_image = np.asarray(transformed_image)

        return GateImage(transformed_image, self.image_width, self.image_height, self.center_x, self.center_y,
                         self.width, self.height)

    def crop(self, x_left_offset: int, x_right_offset: int, y_down_offset: int, y_up_offset: int) -> 'GateImage':
        """
        Crop the image and change all data so that they match to the new image

        :param x_left_offset: x-offset from the left side
        :param x_right_offset: x-offset from the right side
        :param y_down_offset: y-offset from the down
        :param y_up_offset: y-offset from the up
        :return: GateImage object where image is cropped compared to the original image
        """

        center_x, center_y = self.gate_center

        # For safety reasons
        if x_left_offset > center_x:
            x_left_offset = 0
        if self.image_width - x_right_offset < center_x:
            x_right_offset = 0
        if y_down_offset > center_y:
            y_down_offset = 0
        if self.image_height - y_up_offset < center_y:
            y_up_offset = 0

        new_image = self.image[y_down_offset:self.image_height - y_up_offset,
                    x_left_offset:self.image_width - x_right_offset]

        new_height, new_width, _ = new_image.shape
        new_center_x = center_x - x_left_offset
        new_center_y = center_y - y_down_offset

        return GateImage(new_image, new_width, new_height, new_center_x, new_center_y, self.width, self.height)

    def show_gate(self) -> None:
        """
        Show gate location on the image as a way to test implemented reshape / data augmentation techniques

        The function will show:
        1. The image with the gate
        2. The center of the gate (red circle)
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
        location_text = ((self.image_width - textsize[0]) // 2, self.image_height - 20)

        image = cv2.putText(image, text, location_text, font, font_scale, (255, 255, 255), thickness)

        cv2.imshow("Gate Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_image_data(self) -> Tuple[np.ndarray, int, List[int]]:
        """
        Get data needed for training the models

        :return: tuple containing image, gate location as a code and gate coordinates
        """

        gate_coordinates = [self.top_left_corner[0], self.top_left_corner[1], self.bottom_right_corner[0],
                            self.bottom_right_corner[1]]

        return self.image.transpose((2, 0, 1)), self.gate_location.value, gate_coordinates
