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

# Read the image.
image = cv.imread("./data/images/noised_0.jpg", cv.IMREAD_GRAYSCALE)

# Create a filtered image.
filtered_image = np.zeros_like(image)

# Apply the filter manually.
for i in range(1, image.shape[0] - 1):
    for j in range(1, image.shape[1] - 1):
        filtered_image[i, j] = np.median(image[i - 1:i + 2, j - 1:j + 2])

# Show the original image.
cv.imshow("Original Image", image)
cv.imshow("Filtered Image", filtered_image)
cv.waitKey(0)

# Apply the median filter using OpenCV.
filtered_image = cv.medianBlur(image, 3)

# Show the original image.
cv.imshow("Filtered with OpenCV", np.hstack((image, filtered_image)))
cv.waitKey(0)

cv.destroyAllWindows()
