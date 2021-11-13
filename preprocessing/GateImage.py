import numpy as np
import copy
import cv2


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

    def __get_gate_location(self) -> str:
        """
        Get gate location from it's coordinates

        :return: One of five gate locations (fully visible, up, right, down, left)
        """

        if self.bottom_right_corner[0] > self.image_width:
            return "Gate is located to the right"
        elif self.top_left_corner[0] < 0:
            return "Gate is located to the left"
        elif self.bottom_right_corner[1] > self.image_height:
            return "Gate is located to the down"
        elif self.top_left_corner[1] < 0:
            return "Gate is located to the up"
        else:
            return "Gate is fully visible"

    def show_gate(self) -> None:
        """
        Function used primarily for testing.
        Using cv2 this function will show the following things:
        1. The image with the gate
        2. The center of the gate
        3. If possible it will show the location of the gate (red rectangle)
        4. It will print the location of the gate (see __get_gate_location())
        """

        image = cv2.circle(copy.deepcopy(self.image), self.gate_center, 15, (0, 0, 255), -1)
        if self.gate_location == "Gate is fully visible":
            image = cv2.rectangle(image, self.top_left_corner, self.bottom_right_corner, (0, 0, 255), 2)

        # We want to have the descryption centered around x-axis
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2

        textsize = cv2.getTextSize(self.gate_location, font, font_scale, thickness)[0]
        location_text = ((self.image_width - textsize[0]) // 2, self.image_height - 30)

        image = cv2.putText(image, self.gate_location, location_text, font, font_scale, (255, 255, 255), thickness)

        cv2.imshow("Gate Image", image)
        cv2.waitKey(0)

    def flip_image(self) -> 'GateImage':
        """
        Flip image across it's x-axis

        :return: GateImage object where gate is flipped compared to the original GateImage
        """

        new_image = cv2.flip(copy.deepcopy(self.image), 1)
        new_center_x = self.image_width - self.center_x

        return GateImage(new_image, self.image_width, self.image_height, new_center_x, self.center_y, self.width,
                         self.height)

    # TO DO
    # THIS SHOULD RETURN ALL VARIABLES NEEDED FOR TRAINING
    def get_info_image(self):
        return None
