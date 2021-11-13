from preprocessing.GateImage import GateImage
import cv2
import numpy as np

if __name__ == '__main__':
    image = np.array(cv2.imread('data/1.jpg'))
    with open('data/1.txt') as f:
        data = f.readline().split()
        data = list(map(float, data))

    image_height, image_width, _ = image.shape
    data[1] *= image_width
    data[2] *= image_height
    data[3] *= image_width
    data[4] *= image_height
    data = list(map(int, data))

    test = GateImage(image, image_width, image_height, data[1], data[2], data[3], data[4])
    test2 = test.flip_image()
    test2.show_gate()


