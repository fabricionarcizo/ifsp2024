#
# MIT License
#
# Copyright (c) 2024 Fabricio Batista Narcizo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#/

# Import the necessary packages.
import cv2 as cv
import numpy as np


def on_translation_x_change(value):
    global translation_x
    translation_x = value


def on_translation_y_change(value):
    global translation_y
    translation_y = value


def on_scale_change(value):
    if value == 0:
        return
    global scale
    scale = value / 100


def on_size_change(value):
    if value == 0:
        return
    global size
    size = value


def apply_transformations(image):
    rows, cols = image.shape[:2]

    # Translation.
    M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    translated_image = cv.warpAffine(image, M, (cols, rows))

    # Scale.
    scaled_image = cv.resize(translated_image, None, fx=scale, fy=scale)

    # Size.
    resized_image = cv.resize(scaled_image, (size, size))

    return resized_image


# Global variables.
translation_x = 0
translation_y = 0

# Load image.
image = cv.imread("./data/images/lena.jpg")

# Create a window.
cv.namedWindow("Image")

# Create trackbars.
cv.createTrackbar("Translation X", "Image", 0, 100, on_translation_x_change)
cv.createTrackbar("Translation Y", "Image", 0, 100, on_translation_y_change)
cv.createTrackbar("Scale", "Image", 100, 200, on_scale_change)
cv.createTrackbar("Size", "Image", 100, 500, on_size_change)

while True:
    # Apply transformations.
    transformed_image = apply_transformations(image)

    # Show the transformed image.
    cv.imshow("Image", transformed_image)

    # Exit on "q" key press.
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources.
cv.destroyAllWindows()
