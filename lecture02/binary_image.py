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

def on_change(threshold: int) -> None:
    # Create a binary image.
    _, binary_image = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)

    # Show the binary image.
    cv.imshow("Binary", binary_image)

# Read the image.
image = cv.imread("./data/images/lena.jpg", cv.IMREAD_GRAYSCALE)

# Create a window.
cv.namedWindow("Image", cv.WINDOW_NORMAL)

# Create trackbars.
cv.createTrackbar("Threshold", "Image", 0, 255, on_change)

# Show the original image.
cv.imshow("Image", image)
cv.waitKey(0)
