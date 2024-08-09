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

# Load an image from the disk.
image = cv.imread("data/images/spider-man.png")
print(f"Image Resolution: {image.shape}")

# Split the image into its channels using slicing.
image_blue = image[:, :, 0]
image_green = image[:, :, 1]
image_red = image[:, :, 2]

channels = np.hstack((image_blue, image_green, image_red))

# Show the images.
cv.imshow("Image", image)
cv.imshow("Channels", channels)
cv.waitKey(0)

# Split the image into its channels using the split function.
image_blue, image_green, image_red = cv.split(image)

zeros = np.zeros_like(image_blue)

image_blue = cv.merge((image_blue, zeros, zeros))
image_green = cv.merge((zeros, image_green, zeros))
image_red = cv.merge((zeros, zeros, image_red))

channels = np.hstack((image_blue, image_green, image_red))

# Show the images.
cv.imshow("Image", image)
cv.imshow("Channels", channels)
cv.waitKey(0)
